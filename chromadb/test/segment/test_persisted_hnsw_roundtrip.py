from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator
import pickle

import numpy as np
import pytest

from chromadb.api.client import Client
from chromadb.config import Settings, System
from chromadb.db.impl.sqlite import SqliteDB
from chromadb.segment import VectorReader
from chromadb.segment.impl.manager.local import LocalSegmentManager
from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData


PERSISTENT_HNSW_METADATA = {"hnsw:batch_size": 3, "hnsw:sync_threshold": 3}


def _persistent_settings(api_impl: str, persist_directory: str) -> Settings:
    return Settings(
        chroma_api_impl=api_impl,
        chroma_sysdb_impl="chromadb.db.impl.sqlite.SqliteDB",
        chroma_producer_impl="chromadb.db.impl.sqlite.SqliteDB",
        chroma_consumer_impl="chromadb.db.impl.sqlite.SqliteDB",
        chroma_segment_manager_impl="chromadb.segment.impl.manager.local.LocalSegmentManager",
        allow_reset=True,
        is_persistent=True,
        persist_directory=persist_directory,
    )


@contextmanager
def _persistent_system(
    api_impl: str, persist_directory: str
) -> Generator[System, None, None]:
    system = System(_persistent_settings(api_impl, persist_directory))
    system.start()
    try:
        yield system
    finally:
        system.stop()


def _metadata_file(system: System, collection_id: object) -> Path:
    manager = system.instance(LocalSegmentManager)
    segment = manager.get_segment(collection_id, VectorReader)
    return Path(segment._get_metadata_file())  # type: ignore[attr-defined]


def _vector_segment_id(system: System, collection_id: object) -> object:
    manager = system.instance(LocalSegmentManager)
    segment = manager.get_segment(collection_id, VectorReader)
    return segment._id  # type: ignore[attr-defined]


def _vector_segment(system: System, collection_id: object) -> object:
    manager = system.instance(LocalSegmentManager)
    return manager.get_segment(collection_id, VectorReader)


def _require_rust_bindings() -> None:
    pytest.importorskip("chromadb_rust_bindings")


def _assert_single_embedding(collection: object, record_id: str, embedding: list[float]) -> None:
    result = collection.get(ids=[record_id], include=["embeddings"])
    assert result["ids"] == [record_id]
    np.testing.assert_allclose(np.asarray(result["embeddings"][0]), np.asarray(embedding))


def _persist_updated_vector(collection: object) -> None:
    collection.add(ids=["a"], embeddings=[[1.0, 2.0, 3.0]])
    collection.add(ids=["b"], embeddings=[[9.0, 8.0, 7.0]])
    collection.upsert(ids=["a"], embeddings=[[3.0, 2.0, 1.0]])
    collection.delete(ids=["b"])
    collection.upsert(ids=["a"], embeddings=[[3.0, 2.0, 1.0]])
    collection.upsert(ids=["a"], embeddings=[[3.0, 2.0, 1.0]])


def _persist_delete_all(collection: object) -> None:
    collection.add(ids=["gone"], embeddings=[[1.0, 1.0, 1.0]])
    collection.upsert(ids=["gone"], embeddings=[[1.0, 1.0, 1.0]])
    collection.delete(ids=["gone"])


def _assert_metadata_persisted(system: System, collection_id: object) -> Path:
    metadata_file = _metadata_file(system, collection_id)
    assert metadata_file.exists()
    return metadata_file


def _break_hnsw_index_file(system: System, collection_id: object) -> None:
    metadata_file = _metadata_file(system, collection_id)
    index_files = [path for path in metadata_file.parent.iterdir() if path.name != metadata_file.name]
    assert index_files
    index_files[0].unlink()


def _corrupt_hnsw_index_file(system: System, collection_id: object) -> None:
    metadata_file = _metadata_file(system, collection_id)
    index_files = [path for path in metadata_file.parent.iterdir() if path.name != metadata_file.name]
    assert index_files
    index_files[0].write_bytes(b"corrupt hnsw index")


def _set_sqlite_max_seq_id(
    system: System, collection_id: object, seq_id: int | None
) -> None:
    sqlite = system.instance(SqliteDB)
    segment_id = _vector_segment_id(system, collection_id)
    db_segment_id = sqlite.uuid_to_db(segment_id)
    with sqlite.tx() as cur:
        cur.execute("DELETE FROM max_seq_id WHERE segment_id = ?", (db_segment_id,))
        if seq_id is not None:
            cur.execute(
                "INSERT INTO max_seq_id(segment_id, seq_id) VALUES (?, ?)",
                (db_segment_id, seq_id),
            )


def test_python_persisted_hnsw_round_trip_reopens_updated_vectors() -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_round_trip",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)

            metadata_file = _assert_metadata_persisted(system, collection.id)
            data = PersistentData.load_from_file(str(metadata_file))
            assert data.dimensionality == 3
            assert set(data.id_to_label) == {"a"}
            assert data.label_to_id == {data.id_to_label["a"]: "a"}
            assert data.id_to_seq_id == {"a": data.id_to_seq_id["a"]}

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_round_trip")

            _assert_single_embedding(collection, "a", [3.0, 2.0, 1.0])
            query = collection.query(query_embeddings=[[3.0, 2.0, 1.0]], n_results=1)
            assert query["ids"] == [["a"]]


def test_python_persisted_hnsw_round_trip_recovers_fully_deleted_segment() -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_delete_all",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_delete_all(collection)

            metadata_file = _assert_metadata_persisted(system, collection.id)
            data = PersistentData.load_from_file(str(metadata_file))
            assert data.id_to_label == {}
            assert data.label_to_id == {}
            assert data.id_to_seq_id == {}
            assert data.total_elements_added >= 0

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_delete_all")

            assert collection.get(ids=["gone"], include=["embeddings"])["ids"] == []
            collection.add(ids=["replacement"], embeddings=[[9.0, 9.0, 9.0]])
            _assert_single_embedding(collection, "replacement", [9.0, 9.0, 9.0])


def test_python_persisted_hnsw_metadata_writes_are_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_atomic_metadata_write",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)

            metadata_file = _assert_metadata_persisted(system, collection.id)
            original_bytes = metadata_file.read_bytes()
            segment = _vector_segment(system, collection.id)

            def broken_dump(value: object, file: Any, protocol: int) -> None:
                del value, protocol
                file.write(b"corrupt")
                file.flush()
                raise RuntimeError("broken pickle write")

            monkeypatch.setattr(pickle, "dump", broken_dump)

            with pytest.raises(RuntimeError, match="broken pickle write"):
                segment._persist()  # type: ignore[attr-defined]

            assert metadata_file.read_bytes() == original_bytes
            data = PersistentData.load_from_file(str(metadata_file))
            assert data.id_to_label == {"a": data.id_to_label["a"]}
            assert data.label_to_id == {data.id_to_label["a"]: "a"}
            assert data.id_to_seq_id == {"a": data.id_to_seq_id["a"]}


def test_python_reopen_rejects_missing_sqlite_max_seq_id_for_populated_metadata() -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_missing_sqlite_seq_id",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _assert_metadata_persisted(system, collection.id)
            _set_sqlite_max_seq_id(system, collection.id, None)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_missing_sqlite_seq_id")

            with pytest.raises(ValueError, match="no max_seq_id state"):
                collection.get(ids=["a"], include=["embeddings"])


def test_python_reopen_rejects_stale_sqlite_max_seq_id_for_populated_metadata() -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_stale_sqlite_seq_id",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _assert_metadata_persisted(system, collection.id)
            _set_sqlite_max_seq_id(system, collection.id, 0)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_stale_sqlite_seq_id")

            with pytest.raises(ValueError, match="SQLite max_seq_id is smaller"):
                collection.get(ids=["a"], include=["embeddings"])


def test_python_reopen_migrates_legacy_max_seq_id_when_sqlite_state_is_missing() -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_legacy_sqlite_seq_id",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            metadata_file = _assert_metadata_persisted(system, collection.id)
            _set_sqlite_max_seq_id(system, collection.id, None)

            data = PersistentData.load_from_file(str(metadata_file))
            data.max_seq_id = max(data.id_to_seq_id.values())
            with metadata_file.open("wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_legacy_sqlite_seq_id")

            _assert_single_embedding(collection, "a", [3.0, 2.0, 1.0])

            sqlite = system.instance(SqliteDB)
            segment_id = sqlite.uuid_to_db(_vector_segment_id(system, collection.id))
            with sqlite.tx() as cur:
                cur.execute(
                    "SELECT seq_id FROM max_seq_id WHERE segment_id = ?",
                    (segment_id,),
                )
                row = cur.fetchone()
            assert row is not None
            assert row[0] == max(data.id_to_seq_id.values())


def test_python_reopen_rejects_missing_hnsw_index_file() -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_missing_hnsw_file",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _assert_metadata_persisted(system, collection.id)
            _break_hnsw_index_file(system, collection.id)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_missing_hnsw_file")

            with pytest.raises(RuntimeError):
                collection.get(ids=["a"], include=["embeddings"])


def test_python_reopen_rejects_corrupt_hnsw_index_file() -> None:
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_corrupt_hnsw_file",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _assert_metadata_persisted(system, collection.id)
            _corrupt_hnsw_index_file(system, collection.id)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_corrupt_hnsw_file")

            with pytest.raises(RuntimeError):
                collection.get(ids=["a"], include=["embeddings"])


def test_python_persisted_hnsw_written_data_reopens_in_rust_bindings() -> None:
    _require_rust_bindings()
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_to_rust",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _assert_metadata_persisted(system, collection.id)

        with _persistent_system(
            "chromadb.api.rust.RustBindingsAPI", persist_directory
        ) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_to_rust")

            _assert_single_embedding(collection, "a", [3.0, 2.0, 1.0])
            assert collection.query(query_embeddings=[[3.0, 2.0, 1.0]], n_results=1)[
                "ids"
            ] == [["a"]]


def test_python_persisted_hnsw_reopen_in_rust_rejects_missing_sqlite_max_seq_id() -> None:
    _require_rust_bindings()
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_to_rust_missing_sqlite_seq_id",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _set_sqlite_max_seq_id(system, collection.id, None)

        with _persistent_system(
            "chromadb.api.rust.RustBindingsAPI", persist_directory
        ) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_to_rust_missing_sqlite_seq_id")

            with pytest.raises(Exception, match="SQLite max_seq_id is missing"):
                collection.get(ids=["a"], include=["embeddings"])


def test_python_persisted_hnsw_reopen_in_rust_rejects_stale_sqlite_max_seq_id() -> None:
    _require_rust_bindings()
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_to_rust_stale_sqlite_seq_id",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _set_sqlite_max_seq_id(system, collection.id, 0)

        with _persistent_system(
            "chromadb.api.rust.RustBindingsAPI", persist_directory
        ) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_to_rust_stale_sqlite_seq_id")

            with pytest.raises(Exception, match="SQLite max_seq_id is smaller"):
                collection.get(ids=["a"], include=["embeddings"])


def test_legacy_python_metadata_without_dimensionality_reopens_in_rust_bindings() -> None:
    _require_rust_bindings()
    with TemporaryDirectory() as persist_directory:
        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "python_legacy_to_rust",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            collection.add(ids=["a"], embeddings=[[7.0, 8.0, 9.0]])
            collection.upsert(ids=["a"], embeddings=[[7.0, 8.0, 9.0]])
            collection.upsert(ids=["a"], embeddings=[[7.0, 8.0, 9.0]])

            metadata_file = _assert_metadata_persisted(system, collection.id)
            data = PersistentData.load_from_file(str(metadata_file))
            data.dimensionality = None
            with metadata_file.open("wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        with _persistent_system(
            "chromadb.api.rust.RustBindingsAPI", persist_directory
        ) as system:
            client = Client.from_system(system)
            collection = client.get_collection("python_legacy_to_rust")
            _assert_single_embedding(collection, "a", [7.0, 8.0, 9.0])


def test_rust_persisted_hnsw_written_data_reopens_in_python() -> None:
    _require_rust_bindings()
    with TemporaryDirectory() as persist_directory:
        with _persistent_system(
            "chromadb.api.rust.RustBindingsAPI", persist_directory
        ) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "rust_to_python",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("rust_to_python")

            _assert_single_embedding(collection, "a", [3.0, 2.0, 1.0])
            assert collection.query(query_embeddings=[[3.0, 2.0, 1.0]], n_results=1)[
                "ids"
            ] == [["a"]]


def test_rust_persisted_hnsw_reopen_in_python_rejects_missing_sqlite_max_seq_id() -> None:
    _require_rust_bindings()
    with TemporaryDirectory() as persist_directory:
        with _persistent_system(
            "chromadb.api.rust.RustBindingsAPI", persist_directory
        ) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "rust_to_python_missing_sqlite_seq_id",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _set_sqlite_max_seq_id(system, collection.id, None)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("rust_to_python_missing_sqlite_seq_id")

            with pytest.raises(ValueError, match="no max_seq_id state"):
                collection.get(ids=["a"], include=["embeddings"])


def test_rust_persisted_hnsw_reopen_in_python_rejects_stale_sqlite_max_seq_id() -> None:
    _require_rust_bindings()
    with TemporaryDirectory() as persist_directory:
        with _persistent_system(
            "chromadb.api.rust.RustBindingsAPI", persist_directory
        ) as system:
            client = Client.from_system(system)
            collection = client.create_collection(
                "rust_to_python_stale_sqlite_seq_id",
                metadata=PERSISTENT_HNSW_METADATA,
            )
            _persist_updated_vector(collection)
            _set_sqlite_max_seq_id(system, collection.id, 0)

        with _persistent_system("chromadb.api.segment.SegmentAPI", persist_directory) as system:
            client = Client.from_system(system)
            collection = client.get_collection("rust_to_python_stale_sqlite_seq_id")

            with pytest.raises(ValueError, match="SQLite max_seq_id is smaller"):
                collection.get(ids=["a"], include=["embeddings"])
