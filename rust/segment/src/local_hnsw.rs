use std::{
    collections::{BinaryHeap, HashMap},
    io::Write,
    num::NonZeroI32,
    path::Path,
    sync::Arc,
};

use chroma_cache::Weighted;
use chroma_error::{ChromaError, ErrorCodes};
use chroma_index::{HnswIndex, HnswIndexConfig, IndexConfig};
use chroma_sqlite::{db::SqliteDb, table::MaxSeqId};
use chroma_types::{
    operator::RecordMeasure, Chunk, Collection, HnswParametersFromSegmentError, LogRecord,
    Operation, OperationRecord, Segment, SegmentUuid,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sea_query::{Expr, OnConflict, Query, SqliteQueryBuilder};
use sea_query_binder::SqlxBinder;
use serde::{Deserialize, Serialize};
use serde_pickle::{DeOptions, SerOptions};
use sqlx::Row;
use thiserror::Error;

#[allow(dead_code)]
const METADATA_FILE: &str = "index_metadata.pickle";

#[allow(dead_code)]
#[derive(Clone)]
pub struct LocalHnswSegmentReader {
    pub index: LocalHnswIndex,
}

#[derive(Error, Debug)]
pub enum LocalHnswSegmentReaderError {
    #[error("Error opening pickle file: {0}")]
    PickleFileOpenError(#[from] std::io::Error),
    #[error("Error deserializing pickle file: {0}")]
    PickleFileDeserializeError(#[from] serde_pickle::Error),
    #[error("Error loading hnsw index")]
    HnswIndexLoadError,
    #[error("Nothing found on disk")]
    UninitializedSegment,
    #[error("Collection is missing HNSW configuration")]
    MissingHnswConfiguration,
    #[error("Could not parse HNSW configuration: {0}")]
    InvalidHnswConfiguration(#[from] HnswParametersFromSegmentError),
    #[error("Error serializing path to string")]
    PersistPathError,
    #[error("Error finding id")]
    IdNotFound,
    #[error("Error getting embedding")]
    GetEmbeddingError,
    #[error("Error querying knn")]
    QueryError,
    #[error("Error reading from sqlite: {0}")]
    SqliteError(#[from] sqlx::error::Error),
    #[error("Invalid persisted local HNSW metadata: {0}")]
    InvalidPersistedMetadata(String),
}

impl ChromaError for LocalHnswSegmentReaderError {
    fn code(&self) -> ErrorCodes {
        match self {
            LocalHnswSegmentReaderError::PickleFileOpenError(_) => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::PickleFileDeserializeError(_) => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::HnswIndexLoadError => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::UninitializedSegment => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::MissingHnswConfiguration => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::InvalidHnswConfiguration(err) => err.code(),
            LocalHnswSegmentReaderError::PersistPathError => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::IdNotFound => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::GetEmbeddingError => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::QueryError => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::SqliteError(_) => ErrorCodes::Internal,
            LocalHnswSegmentReaderError::InvalidPersistedMetadata(_) => ErrorCodes::Internal,
        }
    }
}

#[derive(Debug)]
enum ValidatedIdMap {
    Uninitialized,
    Initialized {
        id_map: IdMap,
        dimensionality: NonZeroI32,
    },
}

fn validate_persisted_id_map(
    id_map: IdMap,
    expected_dimensionality: usize,
) -> Result<ValidatedIdMap, String> {
    let live_maps_are_empty = id_map.id_to_label.is_empty()
        && id_map.label_to_id.is_empty()
        && id_map.id_to_seq_id.is_empty();
    if live_maps_are_empty {
        return Ok(ValidatedIdMap::Uninitialized);
    }

    if id_map.id_to_label.is_empty()
        || id_map.label_to_id.is_empty()
        || id_map.id_to_seq_id.is_empty()
    {
        return Err("segment has partially populated persisted metadata".to_string());
    }

    if id_map.id_to_label.len() != id_map.label_to_id.len() {
        return Err("persisted label maps are inconsistent".to_string());
    }

    if id_map.id_to_label.len() != id_map.id_to_seq_id.len() {
        return Err("persisted seq id map does not match labels".to_string());
    }

    let mut max_label = 0u32;
    let mut max_persisted_seq_id = 0u32;
    for (user_id, label) in &id_map.id_to_label {
        if *label == 0 {
            return Err("persisted labels must be positive".to_string());
        }

        match id_map.label_to_id.get(label) {
            Some(reverse_user_id) if reverse_user_id == user_id => {}
            _ => return Err("persisted label maps are inconsistent".to_string()),
        }

        let persisted_seq_id = match id_map.id_to_seq_id.get(user_id) {
            Some(seq_id) => *seq_id,
            None => return Err("persisted seq id map does not match labels".to_string()),
        };

        max_label = max_label.max(*label);
        max_persisted_seq_id = max_persisted_seq_id.max(persisted_seq_id);
    }

    if id_map.total_elements_added < max_label {
        return Err("persisted total_elements_added is smaller than its labels".to_string());
    }

    if let Some(max_seq_id) = id_map.max_seq_id {
        if u64::from(max_persisted_seq_id) > max_seq_id {
            return Err("persisted max_seq_id is smaller than its seq ids".to_string());
        }
    }

    let persisted_dimensionality = id_map.dimensionality.unwrap_or(expected_dimensionality);

    if persisted_dimensionality == 0 {
        return Err("segment has persisted labels but dimensionality is 0".to_string());
    }

    if persisted_dimensionality != expected_dimensionality {
        return Err(format!(
            "persisted dimensionality {} does not match collection dimensionality {}",
            persisted_dimensionality, expected_dimensionality
        ));
    }

    let persisted_dimensionality = i32::try_from(persisted_dimensionality).map_err(|_| {
        "persisted dimensionality exceeds the range supported by the HNSW index".to_string()
    })?;
    let persisted_dimensionality = NonZeroI32::new(persisted_dimensionality)
        .expect("persisted dimensionality was already checked to be non-zero");

    Ok(ValidatedIdMap::Initialized {
        id_map,
        dimensionality: persisted_dimensionality,
    })
}

async fn get_current_seq_id(
    segment: &Segment,
    sql_db: &SqliteDb,
) -> Result<Option<u64>, sqlx::error::Error> {
    let (query, values) = Query::select()
        .column(MaxSeqId::SeqId)
        .from(MaxSeqId::Table)
        .and_where(Expr::col(MaxSeqId::SegmentId).eq(segment.id.to_string()))
        .build_sqlx(SqliteQueryBuilder);
    let row = sqlx::query_with(&query, values)
        .fetch_optional(sql_db.get_conn())
        .await?;
    row.map(|row| row.try_get::<u64, _>(0)).transpose()
}

async fn get_or_migrate_current_seq_id(
    segment: &Segment,
    sql_db: &SqliteDb,
    legacy_max_seq_id: Option<u64>,
) -> Result<Option<u64>, sqlx::error::Error> {
    let current_seq_id = get_current_seq_id(segment, sql_db).await?;
    if current_seq_id.is_some() {
        return Ok(current_seq_id);
    }

    if let Some(max_seq_id) = legacy_max_seq_id {
        let id = segment.id.to_string().into();
        let max_id = max_seq_id.into();
        let (query, values) = Query::insert()
            .into_table(MaxSeqId::Table)
            .columns([MaxSeqId::SegmentId, MaxSeqId::SeqId])
            .values([id, max_id])
            .expect("max_seq_id values should build")
            .on_conflict(
                OnConflict::column(MaxSeqId::SegmentId)
                    .do_nothing()
                    .to_owned(),
            )
            .build_sqlx(SqliteQueryBuilder);
        let _ = sqlx::query_with(&query, values)
            .execute(sql_db.get_conn())
            .await?;
        return Ok(Some(max_seq_id));
    }

    Ok(None)
}

fn validate_current_seq_id_state(current_seq_id: Option<u64>, id_map: &IdMap) -> Result<u64, String> {
    let max_persisted_seq_id = id_map.id_to_seq_id.values().copied().max().map(u64::from);

    match current_seq_id {
        Some(current_seq_id) => {
            if let Some(max_persisted_seq_id) = max_persisted_seq_id {
                if current_seq_id < max_persisted_seq_id {
                    return Err(
                        "persisted SQLite max_seq_id is smaller than persisted seq ids"
                            .to_string(),
                    );
                }
            }
            Ok(current_seq_id)
        }
        None => {
            if max_persisted_seq_id.is_some() {
                Err(
                    "persisted metadata has labels but SQLite max_seq_id is missing"
                        .to_string(),
                )
            } else {
                Ok(0)
            }
        }
    }
}

impl LocalHnswSegmentReader {
    pub fn from_index(hnsw_index: LocalHnswIndex) -> Self {
        Self { index: hnsw_index }
    }

    pub async fn from_segment(
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
        persist_root: Option<String>,
        sql_db: SqliteDb,
    ) -> Result<Self, LocalHnswSegmentReaderError> {
        let hnsw_configuration = collection
            .schema
            .as_ref()
            .map(|schema| schema.get_internal_hnsw_config_with_legacy_fallback(segment))
            .transpose()?
            .flatten()
            .ok_or(LocalHnswSegmentReaderError::MissingHnswConfiguration)?;

        match persist_root {
            Some(path_str) => {
                let path = Path::new(&path_str);
                let index_folder = path.join(segment.id.to_string());
                if !index_folder.exists() {
                    // Return uninitialized reader.
                    return Err(LocalHnswSegmentReaderError::UninitializedSegment);
                }
                let index_folder_str = match index_folder.to_str() {
                    Some(path) => path,
                    None => return Err(LocalHnswSegmentReaderError::PersistPathError),
                };
                let pickle_file_path = path.join(segment.id.to_string()).join(METADATA_FILE);
                if pickle_file_path.exists() {
                    let file = tokio::fs::File::open(pickle_file_path)
                        .await?
                        .into_std()
                        .await;
                    let id_map: IdMap = serde_pickle::from_reader(file, DeOptions::new())?;
                    match validate_persisted_id_map(id_map, dimensionality)
                        .map_err(LocalHnswSegmentReaderError::InvalidPersistedMetadata)?
                    {
                        ValidatedIdMap::Initialized {
                            id_map,
                            dimensionality: persisted_dimensionality,
                        } => {
                            // Load hnsw index.
                            let index_config = IndexConfig::new(
                                persisted_dimensionality.get(),
                                hnsw_configuration.space.clone().into(),
                            );
                            let index = HnswIndex::load(
                                index_folder_str,
                                &index_config,
                                hnsw_configuration.ef_search,
                                chroma_index::IndexUuid(segment.id.0),
                            )
                            .map_err(|_| LocalHnswSegmentReaderError::HnswIndexLoadError)?;

                            let current_seq_id =
                                get_or_migrate_current_seq_id(segment, &sql_db, id_map.max_seq_id)
                                    .await?;
                            let current_seq_id =
                                validate_current_seq_id_state(current_seq_id, &id_map).map_err(
                                    LocalHnswSegmentReaderError::InvalidPersistedMetadata,
                                )?;

                            // TODO(Sanket): Set allow reset appropriately.
                            return Ok(Self {
                                index: LocalHnswIndex {
                                    inner: Arc::new(tokio::sync::RwLock::new(Inner {
                                        index,
                                        id_map,
                                        index_init: true,
                                        allow_reset: false,
                                        num_elements_since_last_persist: 0,
                                        last_seen_seq_id: current_seq_id,
                                        sync_threshold: hnsw_configuration.sync_threshold,
                                        persist_path: Some(index_folder_str.to_string()),
                                        sqlite: sql_db,
                                    })),
                                },
                            });
                        }
                        ValidatedIdMap::Uninitialized => {
                            // An empty reader.
                            return Err(LocalHnswSegmentReaderError::UninitializedSegment);
                        }
                    }
                }
                // Return uninitialized reader.
                Err(LocalHnswSegmentReaderError::UninitializedSegment)
            }
            None => {
                let index_config = IndexConfig::new(
                    dimensionality as i32,
                    hnsw_configuration.space.clone().into(),
                );
                let hnsw_config = HnswIndexConfig::new_ephemeral(
                    hnsw_configuration.max_neighbors,
                    hnsw_configuration.ef_construction,
                    hnsw_configuration.ef_search,
                );

                // TODO(Sanket): HnswIndex init is not thread safe. We should not call it from multiple threads
                let index = HnswIndex::init(
                    &index_config,
                    Some(&hnsw_config),
                    chroma_index::IndexUuid(segment.id.0),
                )
                .map_err(|_| LocalHnswSegmentReaderError::HnswIndexLoadError)?;

                Ok(Self {
                    index: LocalHnswIndex {
                        inner: Arc::new(tokio::sync::RwLock::new(Inner {
                            index,
                            id_map: Default::default(),
                            index_init: true,
                            allow_reset: false,
                            num_elements_since_last_persist: 0,
                            last_seen_seq_id: 0,
                            sync_threshold: hnsw_configuration.sync_threshold,
                            persist_path: None,
                            sqlite: sql_db,
                        })),
                    },
                })
            }
        }
    }

    pub async fn get_embedding_by_offset_id(
        &self,
        offset_id: u32,
    ) -> Result<Vec<f32>, LocalHnswSegmentReaderError> {
        let guard = self.index.inner.read().await;
        guard
            .index
            .get(offset_id as usize)
            .map_err(|_| LocalHnswSegmentReaderError::GetEmbeddingError)?
            .ok_or(LocalHnswSegmentReaderError::GetEmbeddingError)
    }

    pub async fn current_max_seq_id(
        &self,
        segment_id: &SegmentUuid,
    ) -> Result<u64, LocalHnswSegmentReaderError> {
        let guard = self.index.inner.read().await;
        let (sql, values) = Query::select()
            .column(MaxSeqId::SeqId)
            .from(MaxSeqId::Table)
            .and_where(Expr::col(MaxSeqId::SegmentId).eq(segment_id.to_string()))
            .build_sqlx(SqliteQueryBuilder);
        let row_opt = sqlx::query_with(&sql, values)
            .fetch_optional(guard.sqlite.get_conn())
            .await?;
        Ok(row_opt
            .map(|row| row.try_get::<u64, _>(0))
            .transpose()?
            .unwrap_or_default())
    }

    pub async fn get_embedding_by_user_id(
        &self,
        user_id: &String,
    ) -> Result<Vec<f32>, LocalHnswSegmentReaderError> {
        let offset_id = self.get_offset_id_by_user_id(user_id).await?;
        self.get_embedding_by_offset_id(offset_id).await
    }

    pub async fn get_offset_id_by_user_id(
        &self,
        user_id: &String,
    ) -> Result<u32, LocalHnswSegmentReaderError> {
        let guard = self.index.inner.read().await;
        guard
            .id_map
            .id_to_label
            .get(user_id)
            .cloned()
            .ok_or(LocalHnswSegmentReaderError::IdNotFound)
    }

    pub async fn get_user_id_by_offset_id(
        &self,
        offset_id: u32,
    ) -> Result<String, LocalHnswSegmentReaderError> {
        let guard = self.index.inner.read().await;
        guard
            .id_map
            .label_to_id
            .get(&offset_id)
            .cloned()
            .ok_or(LocalHnswSegmentReaderError::IdNotFound)
    }

    pub async fn query_embedding(
        &self,
        allowed_offset_ids: &[u32],
        embedding: Vec<f32>,
        k: u32,
    ) -> Result<Vec<RecordMeasure>, LocalHnswSegmentReaderError> {
        let guard = self.index.inner.read().await;
        let len_with_deleted = guard.index.len_with_deleted();
        let actual_len = guard.index.len();

        // Bail if the index is empty
        if actual_len == 0 {
            return Ok(Vec::new());
        }

        let delete_percentage = (len_with_deleted - actual_len) as f32 / len_with_deleted as f32;

        // If the index is small and the delete percentage is high, its quite likely that the index is
        // degraded, so we brute force the search
        // Otherwise search the index normally
        if delete_percentage > 0.2 && actual_len < 100 {
            match guard.index.get_all_ids() {
                Ok((valid_ids, _deleted_ids)) => {
                    let mut max_heap = BinaryHeap::new();
                    let allowed_ids_as_set = allowed_offset_ids
                        .iter()
                        .collect::<std::collections::HashSet<_>>();
                    for curr_id in valid_ids.iter() {
                        if !allowed_ids_as_set.is_empty()
                            && !allowed_ids_as_set.contains(&(*curr_id as u32))
                        {
                            continue;
                        }
                        let curr_embedding = guard.index.get(*curr_id);
                        match curr_embedding {
                            Ok(Some(curr_embedding)) => {
                                let curr_embedding = match guard.index.distance_function {
                                    chroma_distance::DistanceFunction::Cosine => {
                                        chroma_distance::normalize(&curr_embedding)
                                    }
                                    _ => curr_embedding,
                                };
                                let curr_distance = guard
                                    .index
                                    .distance_function
                                    .distance(curr_embedding.as_slice(), embedding.as_slice());
                                if max_heap.len() < k as usize {
                                    max_heap.push(RecordMeasure {
                                        offset_id: *curr_id as u32,
                                        measure: curr_distance,
                                    });
                                } else {
                                    // SAFETY(hammadb): We are sure that the heap has at least one element
                                    // because we insert until we have k elements.
                                    let top = max_heap.peek().unwrap();
                                    if top.measure > curr_distance {
                                        max_heap.pop();
                                        max_heap.push(RecordMeasure {
                                            offset_id: *curr_id as u32,
                                            measure: curr_distance,
                                        });
                                    }
                                }
                            }
                            _ => {
                                return Err(LocalHnswSegmentReaderError::QueryError);
                            }
                        }
                    }
                    Ok(max_heap.into_sorted_vec())
                }
                Err(_) => Err(LocalHnswSegmentReaderError::QueryError),
            }
        } else {
            let allowed_ids = allowed_offset_ids
                .iter()
                .map(|oid| *oid as usize)
                .collect::<Vec<_>>();
            let (offset_ids, distances) = guard
                .index
                .query(&embedding, k as usize, allowed_ids.as_slice(), &[])
                .map_err(|_| LocalHnswSegmentReaderError::QueryError)?;

            Ok(offset_ids
                .into_iter()
                .zip(distances)
                .map(|(offset_id, measure)| RecordMeasure {
                    offset_id: offset_id as u32,
                    measure,
                })
                .collect())
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Default)]
struct IdMap {
    dimensionality: Option<usize>,
    #[serde(default)]
    total_elements_added: u32,
    /// The max_seq_id field is deprecated in favor of the sqlite table
    #[serde(default)]
    max_seq_id: Option<u64>,
    #[serde(default)]
    id_to_label: HashMap<String, u32>,
    #[serde(default)]
    label_to_id: HashMap<u32, String>,
    #[serde(default)]
    id_to_seq_id: HashMap<String, u32>,
}

#[allow(dead_code)]
pub struct Inner {
    index: HnswIndex,
    // Loaded from pickle file.
    id_map: IdMap,
    index_init: bool,
    allow_reset: bool,
    num_elements_since_last_persist: u64,
    last_seen_seq_id: u64,
    sync_threshold: usize,
    persist_path: Option<String>,
    sqlite: SqliteDb,
}

#[derive(Clone)]
pub struct LocalHnswIndex {
    inner: Arc<tokio::sync::RwLock<Inner>>,
}

impl LocalHnswIndex {
    pub async fn close(&self) {
        self.inner.write().await.index.close_fd();
    }
    pub async fn start(&self) {
        self.inner.write().await.index.open_fd();
    }
}

impl Weighted for LocalHnswIndex {
    fn weight(&self) -> usize {
        1
    }
}

#[allow(dead_code)]
pub struct LocalHnswSegmentWriter {
    pub index: LocalHnswIndex,
}

#[derive(Error, Debug)]
pub enum LocalHnswSegmentWriterError {
    #[error("Error creating hnsw config object")]
    HnswConfigError(#[from] Box<chroma_index::HnswIndexConfigError>),
    #[error("Error opening pickle file")]
    PickleFileOpenError(#[from] std::io::Error),
    #[error("Error serializing pickle file")]
    PickleFileSerializeError(#[from] serde_pickle::Error),
    #[error("Error deserializing pickle file")]
    PickleFileDeserializeError(serde_pickle::Error),
    #[error("Error loading hnsw index")]
    HnswIndexLoadError,
    #[error("Nothing found on disk")]
    UninitializedSegment,
    #[error("Collection is missing HNSW configuration")]
    MissingHnswConfiguration,
    #[error("Could not parse HNSW configuration: {0}")]
    InvalidHnswConfiguration(#[from] HnswParametersFromSegmentError),
    #[error("Error creating hnsw index")]
    HnswIndexInitError,
    #[error("Error persisting hnsw index")]
    HnswIndexPersistError,
    #[error("Error applying log chunk")]
    EmbeddingNotFound,
    #[error("Error applying log chunk")]
    HnwsIndexAddError,
    #[error("Error applying log chunk")]
    HnswIndexResizeError,
    #[error("Error applying log chunk")]
    HnswIndexDeleteError,
    #[error("Error converting persistant path to string")]
    PersistPathError,
    #[error("Invalid log offset for persisted HNSW metadata: {0}")]
    InvalidLogOffset(i64),
    #[error("Error updating max sequence id")]
    QueryBuilderError(#[from] sea_query::error::Error),
    #[error("Error updating max sequence id")]
    MaxSeqIdUpdateError(#[from] sqlx::error::Error),
    #[error("Invalid persisted local HNSW metadata: {0}")]
    InvalidPersistedMetadata(String),
}

impl ChromaError for LocalHnswSegmentWriterError {
    fn code(&self) -> ErrorCodes {
        match self {
            LocalHnswSegmentWriterError::HnswConfigError(e) => e.code(),
            LocalHnswSegmentWriterError::PickleFileOpenError(_) => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::PickleFileSerializeError(_) => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::PickleFileDeserializeError(_) => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::HnswIndexLoadError => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::UninitializedSegment => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::MissingHnswConfiguration => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::InvalidHnswConfiguration(err) => err.code(),
            LocalHnswSegmentWriterError::HnswIndexInitError => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::HnswIndexPersistError => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::EmbeddingNotFound => ErrorCodes::InvalidArgument,
            LocalHnswSegmentWriterError::HnwsIndexAddError => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::HnswIndexResizeError => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::HnswIndexDeleteError => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::PersistPathError => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::InvalidLogOffset(_) => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::QueryBuilderError(_) => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::MaxSeqIdUpdateError(_) => ErrorCodes::Internal,
            LocalHnswSegmentWriterError::InvalidPersistedMetadata(_) => ErrorCodes::Internal,
        }
    }
}

fn write_file_atomically<E, F>(path: &Path, write: F) -> Result<(), E>
where
    E: From<std::io::Error>,
    F: FnOnce(&mut std::io::BufWriter<&mut std::fs::File>) -> Result<(), E>,
{
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "atomic write path is missing a parent directory",
        )
    })?;
    let mut temp_file = tempfile::Builder::new()
        .prefix(".index_metadata.")
        .suffix(".tmp")
        .tempfile_in(parent)?;
    {
        let mut buffered_file = std::io::BufWriter::new(temp_file.as_file_mut());
        write(&mut buffered_file)?;
        buffered_file.flush()?;
    }
    temp_file.as_file().sync_all()?;
    temp_file
        .into_temp_path()
        .persist(path)
        .map_err(|err| E::from(err.error))?;
    Ok(())
}

impl LocalHnswSegmentWriter {
    pub fn from_index(hnsw_index: LocalHnswIndex) -> Result<Self, LocalHnswSegmentWriterError> {
        Ok(Self { index: hnsw_index })
    }

    pub async fn from_segment(
        collection: &Collection,
        segment: &Segment,
        dimensionality: usize,
        persist_root: Option<String>,
        sql_db: SqliteDb,
    ) -> Result<Self, LocalHnswSegmentWriterError> {
        let hnsw_configuration = collection
            .schema
            .as_ref()
            .map(|schema| schema.get_internal_hnsw_config_with_legacy_fallback(segment))
            .transpose()?
            .flatten()
            .ok_or(LocalHnswSegmentWriterError::MissingHnswConfiguration)?;

        match persist_root {
            Some(path_str) => {
                let path = Path::new(&path_str);
                let index_folder = path.join(segment.id.to_string());
                if !index_folder.exists() {
                    tokio::fs::create_dir_all(&index_folder).await?;
                }
                let index_folder_str = match index_folder.to_str() {
                    Some(path) => path,
                    None => return Err(LocalHnswSegmentWriterError::PersistPathError),
                };
                let pickle_file_path = path.join(segment.id.to_string()).join(METADATA_FILE);
                if pickle_file_path.exists() {
                    let file = tokio::fs::File::open(pickle_file_path)
                        .await?
                        .into_std()
                        .await;
                    let id_map: IdMap = serde_pickle::from_reader(file, DeOptions::new())?;
                    if let ValidatedIdMap::Initialized {
                        id_map,
                        dimensionality: persisted_dimensionality,
                    } = validate_persisted_id_map(id_map, dimensionality)
                        .map_err(LocalHnswSegmentWriterError::InvalidPersistedMetadata)?
                    {
                        // Load hnsw index.
                        let index_config = IndexConfig::new(
                            persisted_dimensionality.get(),
                            hnsw_configuration.space.clone().into(),
                        );
                        let index = HnswIndex::load(
                            index_folder_str,
                            &index_config,
                            hnsw_configuration.ef_search,
                            chroma_index::IndexUuid(segment.id.0),
                        )
                        .map_err(|_| LocalHnswSegmentWriterError::HnswIndexLoadError)?;

                        let current_seq_id =
                            get_or_migrate_current_seq_id(segment, &sql_db, id_map.max_seq_id)
                                .await?;
                        let current_seq_id =
                            validate_current_seq_id_state(current_seq_id, &id_map).map_err(
                                LocalHnswSegmentWriterError::InvalidPersistedMetadata,
                            )?;

                        // TODO(Sanket): Set allow reset appropriately.
                        return Ok(Self {
                            index: LocalHnswIndex {
                                inner: Arc::new(tokio::sync::RwLock::new(Inner {
                                    index,
                                    id_map,
                                    index_init: true,
                                    allow_reset: false,
                                    num_elements_since_last_persist: 0,
                                    last_seen_seq_id: current_seq_id,
                                    sync_threshold: hnsw_configuration.sync_threshold,
                                    persist_path: Some(index_folder_str.to_string()),
                                    sqlite: sql_db,
                                })),
                            },
                        });
                    }
                }
                // Initialize index.
                let index_config = IndexConfig::new(
                    dimensionality as i32,
                    hnsw_configuration.space.clone().into(),
                );
                let hnsw_config = HnswIndexConfig::new_persistent(
                    hnsw_configuration.max_neighbors,
                    hnsw_configuration.ef_construction,
                    hnsw_configuration.ef_search,
                    &index_folder,
                )?;

                // TODO(Sanket): HnswIndex init is not thread safe. We should not call it from multiple threads
                let index = HnswIndex::init(
                    &index_config,
                    Some(&hnsw_config),
                    chroma_index::IndexUuid(segment.id.0),
                )
                .map_err(|_| LocalHnswSegmentWriterError::HnswIndexInitError)?;
                // Return uninitialized reader.
                Ok(Self {
                    index: LocalHnswIndex {
                        inner: Arc::new(tokio::sync::RwLock::new(Inner {
                            index,
                            id_map: IdMap::default(),
                            index_init: true,
                            allow_reset: false,
                            num_elements_since_last_persist: 0,
                            last_seen_seq_id: 0,
                            sync_threshold: hnsw_configuration.sync_threshold,
                            persist_path: Some(index_folder_str.to_string()),
                            sqlite: sql_db,
                        })),
                    },
                })
            }
            None => {
                let index_config = IndexConfig::new(
                    dimensionality as i32,
                    hnsw_configuration.space.clone().into(),
                );
                let hnsw_config = HnswIndexConfig::new_ephemeral(
                    hnsw_configuration.max_neighbors,
                    hnsw_configuration.ef_construction,
                    hnsw_configuration.ef_search,
                );

                // TODO(Sanket): HnswIndex init is not thread safe. We should not call it from multiple threads
                let index = HnswIndex::init(
                    &index_config,
                    Some(&hnsw_config),
                    chroma_index::IndexUuid(segment.id.0),
                )
                .map_err(|_| LocalHnswSegmentWriterError::HnswIndexInitError)?;
                Ok(Self {
                    index: LocalHnswIndex {
                        inner: Arc::new(tokio::sync::RwLock::new(Inner {
                            index,
                            id_map: Default::default(),
                            index_init: true,
                            allow_reset: false,
                            num_elements_since_last_persist: 0,
                            last_seen_seq_id: 0,
                            sync_threshold: hnsw_configuration.sync_threshold,
                            persist_path: None,
                            sqlite: sql_db,
                        })),
                    },
                })
            }
        }
    }

    // Returns the updated log seq id.
    #[allow(dead_code)]
    pub async fn apply_log_chunk(
        &mut self,
        log_chunk: Chunk<LogRecord>,
    ) -> Result<u32, LocalHnswSegmentWriterError> {
        let mut guard = self.index.inner.write().await;
        let mut next_label = guard.id_map.total_elements_added + 1;
        if log_chunk.is_empty() {
            return Ok(next_label);
        }
        let mut max_seq_id = u64::MIN;
        // In order to insert into hnsw index in parallel, we need to collect all the embeddings
        let mut hnsw_batch: HashMap<u32, Vec<(u32, &OperationRecord)>> =
            HashMap::with_capacity(log_chunk.len());
        for (log, _) in log_chunk.iter() {
            if log.log_offset <= guard.last_seen_seq_id as i64 {
                continue;
            }

            guard.num_elements_since_last_persist += 1;
            max_seq_id = max_seq_id.max(log.log_offset as u64);
            match log.record.operation {
                Operation::BackfillFn => {
                    tracing::warn!("BackfillFn not supported for hnsw index");
                    continue;
                }
                Operation::Add => {
                    // only update if the id is not already present
                    if !guard.id_map.id_to_label.contains_key(&log.record.id) {
                        match &log.record.embedding {
                            Some(_embedding) => {
                                let persisted_seq_id = persistable_log_offset(log.log_offset)?;
                                guard
                                    .id_map
                                    .id_to_label
                                    .insert(log.record.id.clone(), next_label);
                                guard
                                    .id_map
                                    .label_to_id
                                    .insert(next_label, log.record.id.clone());
                                guard
                                    .id_map
                                    .id_to_seq_id
                                    .insert(log.record.id.clone(), persisted_seq_id);
                                let records_for_label = match hnsw_batch.get_mut(&next_label) {
                                    Some(records) => records,
                                    None => {
                                        hnsw_batch.insert(next_label, Vec::new());
                                        // SAFETY: We just inserted the key. We have exclusive access to the map.
                                        hnsw_batch.get_mut(&next_label).unwrap()
                                    }
                                };
                                records_for_label.push((next_label, &log.record));
                                next_label += 1;
                            }
                            None => {
                                return Err(LocalHnswSegmentWriterError::EmbeddingNotFound);
                            }
                        }
                    }
                }
                Operation::Update => {
                    if let Some(label) = guard.id_map.id_to_label.get(&log.record.id).cloned() {
                        if let Some(_embedding) = &log.record.embedding {
                            let persisted_seq_id = persistable_log_offset(log.log_offset)?;
                            guard
                                .id_map
                                .id_to_seq_id
                                .insert(log.record.id.clone(), persisted_seq_id);
                            let records_for_label = match hnsw_batch.get_mut(&label) {
                                Some(records) => records,
                                None => {
                                    hnsw_batch.insert(label, Vec::new());
                                    // SAFETY: We just inserted the key. We have exclusive access to the map.
                                    hnsw_batch.get_mut(&label).unwrap()
                                }
                            };
                            records_for_label.push((label, &log.record));
                        }
                    }
                }
                Operation::Delete => {
                    if let Some(label) = guard.id_map.id_to_label.get(&log.record.id).cloned() {
                        guard.id_map.id_to_label.remove(&log.record.id);
                        guard.id_map.label_to_id.remove(&label);
                        guard.id_map.id_to_seq_id.remove(&log.record.id);
                        let records_for_label = match hnsw_batch.get_mut(&label) {
                            Some(records) => records,
                            None => {
                                hnsw_batch.insert(label, Vec::new());
                                // SAFETY: We just inserted the key. We have exclusive access to the map.
                                hnsw_batch.get_mut(&label).unwrap()
                            }
                        };
                        records_for_label.push((label, &log.record));
                    }
                }
                Operation::Upsert => {
                    let mut update_label = false;
                    let label = match guard.id_map.id_to_label.get(&log.record.id) {
                        Some(label) => *label,
                        None => {
                            update_label = true;
                            next_label
                        }
                    };
                    match &log.record.embedding {
                        Some(_embedding) => {
                            let persisted_seq_id = persistable_log_offset(log.log_offset)?;
                            guard
                                .id_map
                                .id_to_label
                                .insert(log.record.id.clone(), label);
                            guard
                                .id_map
                                .label_to_id
                                .insert(label, log.record.id.clone());
                            guard
                                .id_map
                                .id_to_seq_id
                                .insert(log.record.id.clone(), persisted_seq_id);
                            let records_for_label = match hnsw_batch.get_mut(&label) {
                                Some(records) => records,
                                None => {
                                    hnsw_batch.insert(label, Vec::new());
                                    // SAFETY: We just inserted the key. We have exclusive access to the map.
                                    hnsw_batch.get_mut(&label).unwrap()
                                }
                            };
                            records_for_label.push((label, &log.record));
                            if update_label {
                                next_label += 1;
                            }
                        }
                        None => {
                            return Err(LocalHnswSegmentWriterError::EmbeddingNotFound);
                        }
                    }
                }
            }
        }

        // Add to hnsw index in parallel using rayon.
        // Resize the index if needed
        let index_len = guard.index.len_with_deleted();
        let index_capacity = guard.index.capacity();
        if index_len + hnsw_batch.len() >= index_capacity {
            let needed_capacity = (index_len + hnsw_batch.len()).next_power_of_two();
            guard
                .index
                .resize(needed_capacity)
                .map_err(|_| LocalHnswSegmentWriterError::HnswIndexResizeError)?;
        }
        let index_for_pool = &guard.index;

        hnsw_batch
            .into_par_iter()
            .map(|(_, records)| {
                for (label, log_record) in records {
                    match log_record.operation {
                        Operation::BackfillFn => {
                            continue;
                        }
                        Operation::Add | Operation::Upsert | Operation::Update => {
                            let embedding = log_record.embedding.as_ref().expect(
                                "Add, update or upsert should have an embedding at this point",
                            );
                            match index_for_pool.add(label as usize, embedding) {
                                Ok(_) => {}
                                Err(_e) => {
                                    return Err(LocalHnswSegmentWriterError::HnwsIndexAddError);
                                }
                            }
                        }
                        Operation::Delete => match index_for_pool.delete(label as usize) {
                            Ok(_) => {}
                            Err(_e) => {
                                return Err(LocalHnswSegmentWriterError::HnswIndexDeleteError);
                            }
                        },
                    }
                }
                Ok(())
            })
            .find_any(|result| result.is_err())
            .unwrap_or(Ok(()))?;

        guard.id_map.total_elements_added = next_label - 1;
        if guard.num_elements_since_last_persist >= guard.sync_threshold as u64 {
            guard = persist(guard).await?;
            let id = guard.index.id.to_string().into();
            let max_id = max_seq_id.into();
            // Persist max_seq_id to sqlite.
            let (query, values) = Query::insert()
                .into_table(MaxSeqId::Table)
                .replace()
                .columns([MaxSeqId::SegmentId, MaxSeqId::SeqId])
                .values([id, max_id])?
                .build_sqlx(SqliteQueryBuilder);
            let _ = sqlx::query_with(&query, values)
                .execute(guard.sqlite.get_conn())
                .await?;
            guard.num_elements_since_last_persist = 0;
        }

        guard.last_seen_seq_id = max_seq_id;

        Ok(next_label)
    }
}

fn persistable_index_dimensionality(
    dimensionality: i32,
) -> Result<usize, LocalHnswSegmentWriterError> {
    if dimensionality <= 0 {
        return Err(LocalHnswSegmentWriterError::InvalidPersistedMetadata(
            format!("index dimensionality must be positive, got {dimensionality}"),
        ));
    }
    usize::try_from(dimensionality).map_err(|_| {
        LocalHnswSegmentWriterError::InvalidPersistedMetadata(format!(
            "index dimensionality exceeds usize range: {dimensionality}"
        ))
    })
}

fn persistable_log_offset(log_offset: i64) -> Result<u32, LocalHnswSegmentWriterError> {
    u32::try_from(log_offset).map_err(|_| LocalHnswSegmentWriterError::InvalidLogOffset(log_offset))
}

async fn persist(
    mut guard: tokio::sync::RwLockWriteGuard<'_, Inner>,
) -> Result<tokio::sync::RwLockWriteGuard<'_, Inner>, LocalHnswSegmentWriterError> {
    if let Some(path) = guard.persist_path.clone() {
        // Persist hnsw index.
        guard
            .index
            .save()
            .map_err(|_| LocalHnswSegmentWriterError::HnswIndexPersistError)?;
        // Persist id map.
        guard.id_map.dimensionality = Some(persistable_index_dimensionality(
            guard.index.dimensionality(),
        )?);
        let metadata_file_path = Path::new(&path).join(METADATA_FILE);

        write_file_atomically::<LocalHnswSegmentWriterError, _>(&metadata_file_path, |buffered_file| {
            serde_pickle::to_writer(buffered_file, &guard.id_map, SerOptions::new())?;
            Ok(())
        })?;
    }
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::{
        validate_current_seq_id_state, validate_persisted_id_map, write_file_atomically, DeOptions, IdMap, LocalHnswSegmentReader,
        LocalHnswSegmentReaderError, LocalHnswSegmentWriter, LocalHnswSegmentWriterError,
        SerOptions, ValidatedIdMap, METADATA_FILE,
    };
    use chroma_config::{registry::Registry, Configurable};
    use chroma_sqlite::config::{MigrationHash, MigrationMode, SqliteDBConfig};
    use chroma_sqlite::db::SqliteDb;
    use chroma_sqlite::table::MaxSeqId;
    use chroma_types::{
        test_segment, Chunk, Collection, KnnIndex, LogRecord, Operation, OperationRecord, Schema,
        Segment, SegmentScope,
    };
    use sea_query::{Expr, Query, SqliteQueryBuilder};
    use sea_query_binder::SqlxBinder;
    use serde::Serialize;
    use std::collections::HashMap;
    use std::fs;
    use std::io::{self, Write};
    use tempfile::tempdir;

    fn populated_id_map(dimensionality: Option<usize>) -> IdMap {
        IdMap {
            dimensionality,
            total_elements_added: 1,
            max_seq_id: None,
            id_to_label: HashMap::from([(String::from("a"), 1)]),
            label_to_id: HashMap::from([(1, String::from("a"))]),
            id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
        }
    }

    #[test]
    fn persisted_id_map_accepts_legacy_metadata_without_dimensionality() {
        let id_map = populated_id_map(None);

        match validate_persisted_id_map(id_map, 3).unwrap() {
            ValidatedIdMap::Initialized { dimensionality, .. } => {
                assert_eq!(dimensionality.get(), 3)
            }
            ValidatedIdMap::Uninitialized => panic!("id map should be initialized"),
        }
    }

    #[test]
    fn persisted_id_map_rejects_dimensionality_mismatch() {
        let id_map = populated_id_map(Some(8));

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("does not match"));
    }

    #[test]
    fn persisted_id_map_rejects_zero_dimensionality_when_labels_exist() {
        let id_map = populated_id_map(Some(0));

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("dimensionality is 0"));
    }

    #[test]
    fn persisted_id_map_rejects_dimensionality_outside_hnsw_range() {
        let id_map = populated_id_map(Some(i32::MAX as usize + 1));

        let err = validate_persisted_id_map(id_map, i32::MAX as usize + 1).unwrap_err();
        assert!(err.contains("exceeds the range"));
    }

    #[test]
    fn persisted_id_map_accepts_matching_dimensionality() {
        let id_map = populated_id_map(Some(3));

        match validate_persisted_id_map(id_map, 3).unwrap() {
            ValidatedIdMap::Initialized { dimensionality, .. } => {
                assert_eq!(dimensionality.get(), 3)
            }
            ValidatedIdMap::Uninitialized => panic!("id map should be initialized"),
        }
    }

    #[test]
    fn empty_persisted_id_map_can_stay_uninitialized() {
        let mut id_map = IdMap::default();
        id_map.total_elements_added = 9;

        match validate_persisted_id_map(id_map, 3).unwrap() {
            ValidatedIdMap::Uninitialized => {}
            ValidatedIdMap::Initialized { .. } => panic!("id map should be uninitialized"),
        }
    }

    #[test]
    fn persisted_id_map_accepts_legacy_metadata_after_pickle_deserialization() {
        let pickle = serde_pickle::to_vec(&populated_id_map(None), SerOptions::new()).unwrap();
        let id_map: IdMap =
            serde_pickle::from_slice(&pickle, DeOptions::new()).expect("pickle should deserialize");

        match validate_persisted_id_map(id_map, 3).unwrap() {
            ValidatedIdMap::Initialized { dimensionality, .. } => {
                assert_eq!(dimensionality.get(), 3)
            }
            ValidatedIdMap::Uninitialized => panic!("id map should be initialized"),
        }
    }

    #[test]
    fn persisted_id_map_rejects_partially_populated_metadata() {
        let id_map = IdMap {
            dimensionality: None,
            total_elements_added: 4,
            max_seq_id: None,
            id_to_label: HashMap::new(),
            label_to_id: HashMap::from([(1, String::from("a"))]),
            id_to_seq_id: HashMap::new(),
        };

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("partially populated"));
    }

    #[test]
    fn persisted_id_map_rejects_inconsistent_label_maps() {
        let id_map = IdMap {
            dimensionality: Some(3),
            total_elements_added: 2,
            max_seq_id: None,
            id_to_label: HashMap::from([(String::from("a"), 1)]),
            label_to_id: HashMap::from([(2, String::from("a"))]),
            id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
        };

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("label maps are inconsistent"));
    }

    #[test]
    fn persisted_id_map_rejects_missing_seq_id_entries() {
        let id_map = IdMap {
            dimensionality: Some(3),
            total_elements_added: 2,
            max_seq_id: None,
            id_to_label: HashMap::from([(String::from("a"), 1), (String::from("b"), 2)]),
            label_to_id: HashMap::from([(1, String::from("a")), (2, String::from("b"))]),
            id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
        };

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("seq id map does not match labels"));
    }

    #[test]
    fn persisted_id_map_rejects_total_smaller_than_max_label() {
        let id_map = IdMap {
            dimensionality: Some(3),
            total_elements_added: 1,
            max_seq_id: None,
            id_to_label: HashMap::from([(String::from("a"), 2)]),
            label_to_id: HashMap::from([(2, String::from("a"))]),
            id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
        };

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("total_elements_added is smaller"));
    }

    #[derive(Serialize)]
    struct LegacyIdMapWithoutSeqIds {
        dimensionality: Option<usize>,
        total_elements_added: u32,
        max_seq_id: Option<u64>,
        id_to_label: HashMap<String, u32>,
        label_to_id: HashMap<u32, String>,
    }

    #[derive(Serialize)]
    struct LegacyIdMapWithoutReverseLabels {
        dimensionality: Option<usize>,
        total_elements_added: u32,
        max_seq_id: Option<u64>,
        id_to_label: HashMap<String, u32>,
        id_to_seq_id: HashMap<String, u32>,
    }

    #[derive(Serialize)]
    struct LegacyIdMapWithoutTotalElementsAdded {
        dimensionality: Option<usize>,
        max_seq_id: Option<u64>,
        id_to_label: HashMap<String, u32>,
        label_to_id: HashMap<u32, String>,
        id_to_seq_id: HashMap<String, u32>,
    }

    #[derive(Serialize)]
    struct LegacyIdMapWithoutForwardLabels {
        dimensionality: Option<usize>,
        total_elements_added: u32,
        max_seq_id: Option<u64>,
        label_to_id: HashMap<u32, String>,
        id_to_seq_id: HashMap<String, u32>,
    }

    #[test]
    fn persisted_id_map_defaults_missing_fields_before_validation() {
        let pickle = serde_pickle::to_vec(
            &LegacyIdMapWithoutSeqIds {
                dimensionality: Some(3),
                total_elements_added: 1,
                max_seq_id: None,
                id_to_label: HashMap::from([(String::from("a"), 1)]),
                label_to_id: HashMap::from([(1, String::from("a"))]),
            },
            SerOptions::new(),
        )
        .unwrap();
        let id_map: IdMap =
            serde_pickle::from_slice(&pickle, DeOptions::new()).expect("pickle should deserialize");

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("partially populated"));
    }

    #[test]
    fn persisted_id_map_defaults_missing_reverse_labels_before_validation() {
        let pickle = serde_pickle::to_vec(
            &LegacyIdMapWithoutReverseLabels {
                dimensionality: Some(3),
                total_elements_added: 1,
                max_seq_id: None,
                id_to_label: HashMap::from([(String::from("a"), 1)]),
                id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
            },
            SerOptions::new(),
        )
        .unwrap();
        let id_map: IdMap =
            serde_pickle::from_slice(&pickle, DeOptions::new()).expect("pickle should deserialize");

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("partially populated"));
    }

    #[test]
    fn persisted_id_map_defaults_missing_total_before_validation() {
        let pickle = serde_pickle::to_vec(
            &LegacyIdMapWithoutTotalElementsAdded {
                dimensionality: Some(3),
                max_seq_id: None,
                id_to_label: HashMap::from([(String::from("a"), 1)]),
                label_to_id: HashMap::from([(1, String::from("a"))]),
                id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
            },
            SerOptions::new(),
        )
        .unwrap();
        let id_map: IdMap =
            serde_pickle::from_slice(&pickle, DeOptions::new()).expect("pickle should deserialize");

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("total_elements_added is smaller"));
    }

    #[test]
    fn persisted_id_map_defaults_missing_forward_labels_before_validation() {
        let pickle = serde_pickle::to_vec(
            &LegacyIdMapWithoutForwardLabels {
                dimensionality: Some(3),
                total_elements_added: 1,
                max_seq_id: None,
                label_to_id: HashMap::from([(1, String::from("a"))]),
                id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
            },
            SerOptions::new(),
        )
        .unwrap();
        let id_map: IdMap =
            serde_pickle::from_slice(&pickle, DeOptions::new()).expect("pickle should deserialize");

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("partially populated"));
    }

    #[test]
    fn persisted_id_map_accepts_empty_metadata_with_historical_total() {
        let id_map = IdMap {
            dimensionality: None,
            total_elements_added: 4,
            max_seq_id: None,
            id_to_label: HashMap::new(),
            label_to_id: HashMap::new(),
            id_to_seq_id: HashMap::new(),
        };

        match validate_persisted_id_map(id_map, 3).unwrap() {
            ValidatedIdMap::Uninitialized => {}
            ValidatedIdMap::Initialized { .. } => panic!("id map should be uninitialized"),
        }
    }

    #[test]
    fn persisted_id_map_rejects_zero_labels() {
        let id_map = IdMap {
            dimensionality: Some(3),
            total_elements_added: 1,
            max_seq_id: None,
            id_to_label: HashMap::from([(String::from("a"), 0)]),
            label_to_id: HashMap::from([(0, String::from("a"))]),
            id_to_seq_id: HashMap::from([(String::from("a"), 1)]),
        };

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("labels must be positive"));
    }

    #[test]
    fn persisted_id_map_rejects_extra_seq_id_entries() {
        let id_map = IdMap {
            dimensionality: Some(3),
            total_elements_added: 1,
            max_seq_id: None,
            id_to_label: HashMap::from([(String::from("a"), 1)]),
            label_to_id: HashMap::from([(1, String::from("a"))]),
            id_to_seq_id: HashMap::from([(String::from("b"), 1)]),
        };

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("seq id map does not match labels"));
    }

    #[test]
    fn persisted_id_map_accepts_max_seq_id_covering_seq_ids() {
        let id_map = IdMap {
            dimensionality: Some(3),
            total_elements_added: 1,
            max_seq_id: Some(5),
            id_to_label: HashMap::from([(String::from("a"), 1)]),
            label_to_id: HashMap::from([(1, String::from("a"))]),
            id_to_seq_id: HashMap::from([(String::from("a"), 3)]),
        };

        match validate_persisted_id_map(id_map, 3).unwrap() {
            ValidatedIdMap::Initialized { .. } => {}
            ValidatedIdMap::Uninitialized => panic!("id map should be initialized"),
        }
    }

    #[test]
    fn persisted_id_map_rejects_max_seq_id_smaller_than_seq_ids() {
        let id_map = IdMap {
            dimensionality: Some(3),
            total_elements_added: 1,
            max_seq_id: Some(2),
            id_to_label: HashMap::from([(String::from("a"), 1)]),
            label_to_id: HashMap::from([(1, String::from("a"))]),
            id_to_seq_id: HashMap::from([(String::from("a"), 3)]),
        };

        let err = validate_persisted_id_map(id_map, 3).unwrap_err();
        assert!(err.contains("max_seq_id is smaller"));
    }

    #[test]
    fn current_seq_id_state_rejects_missing_sqlite_state_for_populated_metadata() {
        let err = validate_current_seq_id_state(None, &populated_id_map(Some(3))).unwrap_err();
        assert!(err.contains("SQLite max_seq_id is missing"));
    }

    #[test]
    fn current_seq_id_state_rejects_stale_sqlite_state_for_populated_metadata() {
        let err =
            validate_current_seq_id_state(Some(0), &populated_id_map(Some(3))).unwrap_err();
        assert!(err.contains("SQLite max_seq_id is smaller"));
    }

    #[test]
    fn current_seq_id_state_accepts_sqlite_state_covering_persisted_seq_ids() {
        assert_eq!(
            validate_current_seq_id_state(Some(3), &populated_id_map(Some(3))).unwrap(),
            3
        );
    }

    #[test]
    fn persistable_index_dimensionality_rejects_non_positive_values() {
        assert!(super::persistable_index_dimensionality(0).is_err());
        assert!(super::persistable_index_dimensionality(-1).is_err());
    }

    #[test]
    fn persistable_index_dimensionality_accepts_positive_values() {
        assert_eq!(super::persistable_index_dimensionality(384).unwrap(), 384);
    }

    #[test]
    fn persistable_log_offset_rejects_negative_values() {
        assert!(matches!(
            super::persistable_log_offset(-1),
            Err(LocalHnswSegmentWriterError::InvalidLogOffset(-1))
        ));
    }

    fn log_record(
        id: &str,
        log_offset: i64,
        operation: Operation,
        embedding: Option<Vec<f32>>,
    ) -> LogRecord {
        LogRecord {
            log_offset,
            record: OperationRecord {
                id: id.to_string(),
                embedding,
                encoding: None,
                metadata: None,
                document: None,
                operation,
            },
        }
    }

    fn test_collection_and_segment() -> (Collection, Segment) {
        let mut collection = Collection::test_collection(3);
        collection.schema = Some(Schema::new_default(KnnIndex::Hnsw));
        let segment = test_segment(collection.collection_id, SegmentScope::VECTOR);
        (collection, segment)
    }

    fn persist_id_map(id_map: &IdMap, segment: &Segment) -> tempfile::TempDir {
        let persist_root = tempdir().expect("persist root should be created");
        let index_folder = persist_root.path().join(segment.id.to_string());
        fs::create_dir(&index_folder).expect("index folder should be created");
        let metadata_file = index_folder.join(METADATA_FILE);
        let mut file = fs::File::create(metadata_file).expect("metadata file should be created");
        serde_pickle::to_writer(&mut file, id_map, SerOptions::new())
            .expect("id map should serialize");
        persist_root
    }

    async fn new_sqlite_db() -> SqliteDb {
        let db_dir = tempdir().expect("sqlite temp dir should be created").keep();
        let db_path = db_dir.join("chroma.sqlite3");
        let config = SqliteDBConfig {
            url: Some(db_path.to_string_lossy().into_owned()),
            hash_type: MigrationHash::MD5,
            migration_mode: MigrationMode::Apply,
        };
        SqliteDb::try_from_config(&config, &Registry::new())
            .await
            .expect("sqlite db should be created")
    }

    async fn set_current_seq_id(sqlite: &SqliteDb, segment: &Segment, seq_id: Option<u64>) {
        let (delete_query, delete_values) = Query::delete()
            .from_table(MaxSeqId::Table)
            .and_where(Expr::col(MaxSeqId::SegmentId).eq(segment.id.to_string()))
            .build_sqlx(SqliteQueryBuilder);
        sqlx::query_with(&delete_query, delete_values)
            .execute(sqlite.get_conn())
            .await
            .expect("max_seq_id row should be deleted");

        if let Some(seq_id) = seq_id {
            let (insert_query, insert_values) = Query::insert()
                .into_table(MaxSeqId::Table)
                .columns([MaxSeqId::SegmentId, MaxSeqId::SeqId])
                .values([segment.id.to_string().into(), seq_id.into()])
                .expect("max_seq_id values should build")
                .build_sqlx(SqliteQueryBuilder);
            sqlx::query_with(&insert_query, insert_values)
                .execute(sqlite.get_conn())
                .await
                .expect("max_seq_id row should be inserted");
        }
    }

    fn remove_one_index_file(persist_root: &tempfile::TempDir, segment: &Segment) {
        let index_folder = persist_root.path().join(segment.id.to_string());
        let mut removed = false;
        for entry in fs::read_dir(index_folder).expect("index folder should be readable") {
            let entry = entry.expect("directory entry should be readable");
            if entry.file_name() != METADATA_FILE {
                fs::remove_file(entry.path()).expect("index file should be removable");
                removed = true;
                break;
            }
        }
        assert!(removed, "an hnsw index file should have been removed");
    }

    fn corrupt_one_index_file(persist_root: &tempfile::TempDir, segment: &Segment) {
        let index_folder = persist_root.path().join(segment.id.to_string());
        let mut corrupted = false;
        for entry in fs::read_dir(index_folder).expect("index folder should be readable") {
            let entry = entry.expect("directory entry should be readable");
            if entry.file_name() != METADATA_FILE {
                fs::write(entry.path(), b"corrupt hnsw index")
                    .expect("index file should be writable");
                corrupted = true;
                break;
            }
        }
        assert!(corrupted, "an hnsw index file should have been corrupted");
    }

    #[test]
    fn atomic_metadata_write_preserves_existing_file_on_failure() {
        let temp_dir = tempdir().expect("temp dir should be created");
        let metadata_file = temp_dir.path().join(METADATA_FILE);
        fs::write(&metadata_file, b"stable metadata").expect("metadata file should be written");

        let result = write_file_atomically::<io::Error, _>(&metadata_file, |buffered_file| {
            buffered_file.write_all(b"corrupt metadata")?;
            Err(io::Error::other("broken atomic write"))
        });

        assert!(result.is_err());
        assert_eq!(
            fs::read(&metadata_file).expect("metadata file should still be readable"),
            b"stable metadata"
        );
    }

    #[tokio::test]
    async fn reader_rejects_invalid_persisted_metadata_before_loading_hnsw_index() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = persist_id_map(&populated_id_map(Some(0)), &segment);
        let sqlite = new_sqlite_db().await;

        let result = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root.path().to_string_lossy().into_owned()),
            sqlite,
        )
        .await;

        let err = match result {
            Err(err) => err,
            Ok(_) => panic!("invalid metadata should be rejected"),
        };
        assert!(matches!(
            err,
            LocalHnswSegmentReaderError::InvalidPersistedMetadata(message)
                if message.contains("dimensionality is 0")
        ));
    }

    #[tokio::test]
    async fn writer_rejects_invalid_persisted_metadata_before_loading_hnsw_index() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = persist_id_map(&populated_id_map(Some(0)), &segment);
        let sqlite = new_sqlite_db().await;

        let result = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root.path().to_string_lossy().into_owned()),
            sqlite,
        )
        .await;

        let err = match result {
            Err(err) => err,
            Ok(_) => panic!("invalid metadata should be rejected"),
        };
        assert!(matches!(
            err,
            LocalHnswSegmentWriterError::InvalidPersistedMetadata(message)
                if message.contains("dimensionality is 0")
        ));
    }

    #[tokio::test]
    async fn reader_reports_deserialize_error_for_truncated_pickle() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let index_folder = persist_root.path().join(segment.id.to_string());
        fs::create_dir(&index_folder).expect("index folder should be created");
        fs::write(index_folder.join(METADATA_FILE), [0x80, 0x03, b'}'])
            .expect("metadata file should be written");
        let sqlite = new_sqlite_db().await;

        let result = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root.path().to_string_lossy().into_owned()),
            sqlite,
        )
        .await;

        assert!(matches!(
            result,
            Err(LocalHnswSegmentReaderError::PickleFileDeserializeError(_))
        ));
    }

    #[tokio::test]
    async fn writer_tracks_id_to_seq_id_across_mutations() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root.path().to_string_lossy().into_owned()),
            sqlite,
        )
        .await
        .expect("writer should initialize");

        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    1,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should succeed");
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    2,
                    Operation::Update,
                    Some(vec![3.0, 2.0, 1.0]),
                )]
                .into(),
            ))
            .await
            .expect("update should succeed");
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "b",
                    3,
                    Operation::Upsert,
                    Some(vec![4.0, 5.0, 6.0]),
                )]
                .into(),
            ))
            .await
            .expect("upsert should succeed");
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record("a", 4, Operation::Delete, None)].into(),
            ))
            .await
            .expect("delete should succeed");

        let guard = writer.index.inner.read().await;
        assert_eq!(guard.id_map.id_to_seq_id.get("b"), Some(&3));
        assert!(!guard.id_map.id_to_seq_id.contains_key("a"));
        assert_eq!(
            guard.id_map.id_to_label.len(),
            guard.id_map.id_to_seq_id.len()
        );
    }

    #[tokio::test]
    async fn reader_reopens_index_persisted_by_writer() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let persist_root_str = persist_root.path().to_string_lossy().into_owned();
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should initialize");
        writer.index.inner.write().await.sync_threshold = 1;

        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    1,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should persist");

        drop(writer);

        let reader = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str),
            sqlite,
        )
        .await
        .expect("reader should reopen persisted index");

        assert_eq!(
            reader
                .get_embedding_by_user_id(&String::from("a"))
                .await
                .expect("embedding should be readable after reopen"),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[tokio::test]
    async fn writer_recovers_from_fully_deleted_persisted_metadata() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let persist_root_str = persist_root.path().to_string_lossy().into_owned();
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should initialize");
        writer.index.inner.write().await.sync_threshold = 1;

        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    1,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should persist");
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record("a", 2, Operation::Delete, None)].into(),
            ))
            .await
            .expect("delete should persist");
        drop(writer);

        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should reopen fully deleted metadata");
        writer.index.inner.write().await.sync_threshold = 1;
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "replacement",
                    3,
                    Operation::Add,
                    Some(vec![9.0, 8.0, 7.0]),
                )]
                .into(),
            ))
            .await
            .expect("replacement add should persist");
        drop(writer);

        let reader = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str),
            sqlite,
        )
        .await
        .expect("reader should reopen replacement index");

        assert_eq!(
            reader
                .get_embedding_by_user_id(&String::from("replacement"))
                .await
                .expect("replacement embedding should be readable after reopen"),
            vec![9.0, 8.0, 7.0]
        );
    }

    #[tokio::test]
    async fn reader_rejects_missing_sqlite_max_seq_id_for_persisted_metadata() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let persist_root_str = persist_root.path().to_string_lossy().into_owned();
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should initialize");
        writer.index.inner.write().await.sync_threshold = 1;
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    1,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should persist");
        drop(writer);

        set_current_seq_id(&sqlite, &segment, None).await;

        let result = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str),
            sqlite,
        )
        .await;

        assert!(matches!(
            result,
            Err(LocalHnswSegmentReaderError::InvalidPersistedMetadata(message))
                if message.contains("SQLite max_seq_id is missing")
        ));
    }

    #[tokio::test]
    async fn reader_rejects_stale_sqlite_max_seq_id_for_persisted_metadata() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let persist_root_str = persist_root.path().to_string_lossy().into_owned();
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should initialize");
        writer.index.inner.write().await.sync_threshold = 1;
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    3,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should persist");
        drop(writer);

        set_current_seq_id(&sqlite, &segment, Some(0)).await;

        let result = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str),
            sqlite,
        )
        .await;

        assert!(matches!(
            result,
            Err(LocalHnswSegmentReaderError::InvalidPersistedMetadata(message))
                if message.contains("SQLite max_seq_id is smaller")
        ));
    }

    #[tokio::test]
    async fn reader_reopens_with_migrated_legacy_max_seq_id() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let persist_root_str = persist_root.path().to_string_lossy().into_owned();
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should initialize");
        writer.index.inner.write().await.sync_threshold = 1;
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    3,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should persist");
        drop(writer);

        let index_folder = persist_root.path().join(segment.id.to_string());
        let metadata_file = index_folder.join(METADATA_FILE);
        let file = fs::File::open(&metadata_file).expect("metadata file should open");
        let mut id_map: IdMap =
            serde_pickle::from_reader(file, DeOptions::new()).expect("id map should deserialize");
        id_map.max_seq_id = Some(3);
        let mut rewritten = fs::File::create(&metadata_file).expect("metadata file should rewrite");
        serde_pickle::to_writer(&mut rewritten, &id_map, SerOptions::new())
            .expect("id map should serialize");

        set_current_seq_id(&sqlite, &segment, None).await;

        let reader = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str),
            sqlite.clone(),
        )
        .await
        .expect("reader should reopen with migrated legacy max_seq_id");

        assert_eq!(
            reader
                .current_max_seq_id(&segment.id)
                .await
                .expect("current max seq id should be readable"),
            3
        );
    }

    #[tokio::test]
    async fn reader_rejects_missing_hnsw_index_file() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let persist_root_str = persist_root.path().to_string_lossy().into_owned();
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should initialize");
        writer.index.inner.write().await.sync_threshold = 1;
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    1,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should persist");
        drop(writer);

        remove_one_index_file(&persist_root, &segment);

        let result = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str),
            sqlite,
        )
        .await;

        assert!(matches!(
            result,
            Err(LocalHnswSegmentReaderError::HnswIndexLoadError)
        ));
    }

    #[tokio::test]
    async fn reader_rejects_corrupt_hnsw_index_file() {
        let (collection, segment) = test_collection_and_segment();
        let persist_root = tempdir().expect("persist root should be created");
        let persist_root_str = persist_root.path().to_string_lossy().into_owned();
        let sqlite = new_sqlite_db().await;
        let mut writer = LocalHnswSegmentWriter::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str.clone()),
            sqlite.clone(),
        )
        .await
        .expect("writer should initialize");
        writer.index.inner.write().await.sync_threshold = 1;
        writer
            .apply_log_chunk(Chunk::new(
                vec![log_record(
                    "a",
                    1,
                    Operation::Add,
                    Some(vec![1.0, 2.0, 3.0]),
                )]
                .into(),
            ))
            .await
            .expect("add should persist");
        drop(writer);

        corrupt_one_index_file(&persist_root, &segment);

        let result = LocalHnswSegmentReader::from_segment(
            &collection,
            &segment,
            3,
            Some(persist_root_str),
            sqlite,
        )
        .await;

        assert!(matches!(
            result,
            Err(LocalHnswSegmentReaderError::HnswIndexLoadError)
        ));
    }
}
