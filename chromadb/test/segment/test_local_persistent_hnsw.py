import pickle

import pytest

from chromadb.segment.impl.vector.local_persistent_hnsw import (
    PersistentData,
    _resolve_current_max_seq_id,
    _validate_persisted_data,
)


def _persistent_data(dimensionality):
    return PersistentData(
        dimensionality=dimensionality,
        total_elements_added=1,
        id_to_label={"a": 1},
        label_to_id={1: "a"},
        id_to_seq_id={"a": 1},
    )


@pytest.mark.parametrize("dimensionality", [None, 0, -1, True, 1.5, "3"])
def test_validate_persisted_data_rejects_invalid_dimensionality_when_labels_exist(
    dimensionality,
):
    with pytest.raises(ValueError, match="dimensionality"):
        _validate_persisted_data(_persistent_data(dimensionality))


@pytest.mark.parametrize("dimensionality", [1, 3, 384])
def test_validate_persisted_data_allows_valid_dimensionality_when_labels_exist(
    dimensionality,
):
    _validate_persisted_data(_persistent_data(dimensionality))


def test_validate_persisted_data_uses_expected_dimensionality_for_legacy_metadata():
    data = _persistent_data(None)

    _validate_persisted_data(data, expected_dimensionality=5)

    assert data.dimensionality == 5


def test_validate_persisted_data_rejects_mismatched_expected_dimensionality():
    with pytest.raises(ValueError, match="does not match the collection dimensionality"):
        _validate_persisted_data(_persistent_data(5), expected_dimensionality=3)


@pytest.mark.parametrize("dimensionality,total_elements_added", [(None, 0), (0, 4), (-1, 9)])
def test_validate_persisted_data_allows_empty_label_map_with_historical_total(
    dimensionality, total_elements_added
):
    data = PersistentData(
        dimensionality=dimensionality,
        total_elements_added=total_elements_added,
        id_to_label={},
        label_to_id={},
        id_to_seq_id={},
    )

    _validate_persisted_data(data)


@pytest.mark.parametrize(
    "label_to_id,id_to_seq_id",
    [
        ({1: "a"}, {}),
        ({}, {"a": 1}),
        ({1: "a"}, {"a": 1}),
    ],
)
def test_validate_persisted_data_rejects_partially_populated_empty_metadata(
    label_to_id, id_to_seq_id
):
    data = PersistentData(
        dimensionality=None,
        total_elements_added=3,
        id_to_label={},
        label_to_id=label_to_id,
        id_to_seq_id=id_to_seq_id,
    )

    with pytest.raises(ValueError, match="partially populated"):
        _validate_persisted_data(data)


def test_validate_persisted_data_rejects_inconsistent_label_maps():
    data = PersistentData(
        dimensionality=3,
        total_elements_added=2,
        id_to_label={"a": 1},
        label_to_id={2: "a"},
        id_to_seq_id={"a": 1},
    )

    with pytest.raises(ValueError, match="label maps are inconsistent"):
        _validate_persisted_data(data)


@pytest.mark.parametrize(
    "attr,value,match",
    [
        ("id_to_label", [], "invalid id_to_label"),
        ("label_to_id", [], "invalid label_to_id"),
        ("id_to_seq_id", [], "invalid id_to_seq_id"),
    ],
)
def test_validate_persisted_data_rejects_non_mapping_fields(attr, value, match):
    data = _persistent_data(3)
    setattr(data, attr, value)

    with pytest.raises(ValueError, match=match):
        _validate_persisted_data(data)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda data: data.id_to_label.__setitem__(1, 1), "invalid id_to_label"),
        (
            lambda data: data.label_to_id.__setitem__(1, 2),
            "invalid label_to_id",
        ),
        (lambda data: data.id_to_seq_id.__setitem__(1, 1), "invalid id_to_seq_id"),
    ],
)
def test_validate_persisted_data_rejects_invalid_map_key_or_value_types(mutate, match):
    data = _persistent_data(3)
    mutate(data)

    with pytest.raises(ValueError, match=match):
        _validate_persisted_data(data)


@pytest.mark.parametrize("label", [0, -1, True, 1.5, "1"])
def test_validate_persisted_data_rejects_invalid_labels(label):
    data = PersistentData(
        dimensionality=3,
        total_elements_added=2,
        id_to_label={"a": label},
        label_to_id={1: "a"},
        id_to_seq_id={"a": 1},
    )

    with pytest.raises(ValueError, match="invalid label"):
        _validate_persisted_data(data)


@pytest.mark.parametrize("seq_id", [-1, True, 1.5, "1"])
def test_validate_persisted_data_rejects_invalid_seq_ids(seq_id):
    data = PersistentData(
        dimensionality=3,
        total_elements_added=2,
        id_to_label={"a": 1},
        label_to_id={1: "a"},
        id_to_seq_id={"a": seq_id},
    )

    with pytest.raises(ValueError, match="invalid seq id"):
        _validate_persisted_data(data)


def test_validate_persisted_data_rejects_missing_seq_id_entries():
    data = PersistentData(
        dimensionality=3,
        total_elements_added=2,
        id_to_label={"a": 1},
        label_to_id={1: "a"},
        id_to_seq_id={},
    )

    with pytest.raises(ValueError, match="seq id map does not match labels"):
        _validate_persisted_data(data)


@pytest.mark.parametrize("max_seq_id", [-1, True, 1.5, "1"])
def test_validate_persisted_data_rejects_invalid_legacy_max_seq_id(max_seq_id):
    data = _persistent_data(3)
    data.max_seq_id = max_seq_id

    with pytest.raises(ValueError, match="invalid max_seq_id"):
        _validate_persisted_data(data)


def test_validate_persisted_data_rejects_legacy_max_seq_id_smaller_than_seq_ids():
    data = _persistent_data(3)
    data.max_seq_id = 0

    with pytest.raises(ValueError, match="max_seq_id is smaller"):
        _validate_persisted_data(data)


@pytest.mark.parametrize("attr", ["id_to_label", "label_to_id", "id_to_seq_id", "total_elements_added"])
def test_validate_persisted_data_rejects_missing_required_attributes(attr):
    data = _persistent_data(3)
    delattr(data, attr)

    with pytest.raises(ValueError, match=f"missing {attr}"):
        _validate_persisted_data(data)


def test_resolve_current_max_seq_id_uses_sqlite_state_when_present():
    data = _persistent_data(3)

    assert _resolve_current_max_seq_id(data, sqlite_max_seq_id=7, default_seq_id=-1) == 7


def test_resolve_current_max_seq_id_rejects_stale_sqlite_state():
    data = _persistent_data(3)

    with pytest.raises(ValueError, match="SQLite max_seq_id is smaller"):
        _resolve_current_max_seq_id(data, sqlite_max_seq_id=0, default_seq_id=-1)


def test_resolve_current_max_seq_id_uses_legacy_state_when_sqlite_missing():
    data = _persistent_data(3)
    data.max_seq_id = 5

    assert _resolve_current_max_seq_id(data, sqlite_max_seq_id=None, default_seq_id=-1) == 5


def test_resolve_current_max_seq_id_rejects_populated_metadata_without_seq_id_state():
    data = _persistent_data(3)

    with pytest.raises(ValueError, match="no max_seq_id state"):
        _resolve_current_max_seq_id(data, sqlite_max_seq_id=None, default_seq_id=-1)


@pytest.mark.parametrize("total_elements_added", [-1, True, 1.5, "1"])
def test_validate_persisted_data_rejects_invalid_historical_total(total_elements_added):
    data = PersistentData(
        dimensionality=3,
        total_elements_added=total_elements_added,
        id_to_label={"a": 1},
        label_to_id={1: "a"},
        id_to_seq_id={"a": 1},
    )

    with pytest.raises(ValueError, match="invalid total_elements_added"):
        _validate_persisted_data(data)


def test_validate_persisted_data_rejects_total_smaller_than_max_label():
    data = PersistentData(
        dimensionality=3,
        total_elements_added=1,
        id_to_label={"a": 2},
        label_to_id={2: "a"},
        id_to_seq_id={"a": 1},
    )

    with pytest.raises(ValueError, match="total_elements_added is smaller"):
        _validate_persisted_data(data)


@pytest.mark.parametrize("dimensionality", [None, 0, -1, True, 1.5, "3"])
def test_load_from_file_rejects_invalid_dimensionality_when_labels_exist(
    tmp_path, dimensionality
):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(_persistent_data(dimensionality), f, pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="dimensionality"):
        PersistentData.load_from_file(str(path))


def test_load_from_file_allows_empty_label_map_without_dimensionality(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(
            PersistentData(
                dimensionality=None,
                total_elements_added=0,
                id_to_label={},
                label_to_id={},
                id_to_seq_id={},
            ),
            f,
            pickle.HIGHEST_PROTOCOL,
        )

    loaded = PersistentData.load_from_file(str(path))
    assert loaded.dimensionality is None
    assert loaded.id_to_label == {}


def test_load_from_file_rejects_invalid_total_elements_added_with_empty_maps(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(
            PersistentData(
                dimensionality=None,
                total_elements_added=-1,
                id_to_label={},
                label_to_id={},
                id_to_seq_id={},
            ),
            f,
            pickle.HIGHEST_PROTOCOL,
        )

    with pytest.raises(ValueError, match="invalid total_elements_added"):
        PersistentData.load_from_file(str(path))


def test_load_from_file_allows_valid_dimensionality_when_labels_exist(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(
            PersistentData(
                dimensionality=5,
                total_elements_added=1,
                id_to_label={"a": 1},
                label_to_id={1: "a"},
                id_to_seq_id={"a": 1},
            ),
            f,
            pickle.HIGHEST_PROTOCOL,
        )

    loaded = PersistentData.load_from_file(str(path))
    assert loaded.dimensionality == 5
    assert loaded.id_to_label == {"a": 1}


def test_load_from_file_uses_expected_dimensionality_for_legacy_metadata(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(_persistent_data(None), f, pickle.HIGHEST_PROTOCOL)

    loaded = PersistentData.load_from_file(str(path), expected_dimensionality=7)

    assert loaded.dimensionality == 7
    assert loaded.id_to_label == {"a": 1}


def test_load_from_file_rejects_mismatched_expected_dimensionality(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(_persistent_data(7), f, pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="does not match the collection dimensionality"):
        PersistentData.load_from_file(str(path), expected_dimensionality=3)


def test_load_from_file_rejects_inconsistent_metadata(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(
            PersistentData(
                dimensionality=3,
                total_elements_added=1,
                id_to_label={"a": 1},
                label_to_id={1: "b"},
                id_to_seq_id={"a": 1},
            ),
            f,
            pickle.HIGHEST_PROTOCOL,
        )

    with pytest.raises(ValueError, match="label maps are inconsistent"):
        PersistentData.load_from_file(str(path))


def test_load_from_file_uses_expected_dimensionality_when_attr_is_missing(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    data = _persistent_data(3)
    delattr(data, "dimensionality")
    with path.open("wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    loaded = PersistentData.load_from_file(str(path), expected_dimensionality=7)
    assert loaded.dimensionality == 7


@pytest.mark.parametrize("attr", ["id_to_label", "label_to_id", "id_to_seq_id", "total_elements_added"])
def test_load_from_file_rejects_missing_required_attributes(tmp_path, attr):
    path = tmp_path / "index_metadata.pickle"
    data = _persistent_data(3)
    delattr(data, attr)
    with path.open("wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match=f"missing {attr}"):
        PersistentData.load_from_file(str(path))


def test_load_from_file_rejects_invalid_total_elements_added(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(
            PersistentData(
                dimensionality=3,
                total_elements_added=-1,
                id_to_label={"a": 1},
                label_to_id={1: "a"},
                id_to_seq_id={"a": 1},
            ),
            f,
            pickle.HIGHEST_PROTOCOL,
        )

    with pytest.raises(ValueError, match="invalid total_elements_added"):
        PersistentData.load_from_file(str(path))


def test_load_from_file_rejects_legacy_max_seq_id_smaller_than_seq_ids(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    data = _persistent_data(3)
    data.max_seq_id = 0
    with path.open("wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="max_seq_id is smaller"):
        PersistentData.load_from_file(str(path))


def test_load_from_file_rejects_truncated_pickle(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    payload = pickle.dumps(_persistent_data(3), pickle.HIGHEST_PROTOCOL)
    path.write_bytes(payload[:-1])

    with pytest.raises((pickle.UnpicklingError, EOFError)):
        PersistentData.load_from_file(str(path))
