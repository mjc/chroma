import io
import os
import pickle

import pytest

from chromadb.segment.impl.vector.local_persistent_hnsw import (
    PersistentData,
    SafeUnpickler,
)


def _persistent_data(dimensionality):
    return PersistentData(
        dimensionality=dimensionality,
        total_elements_added=10,
        id_to_label={"abc": 1, "def": 2},
        label_to_id={1: "abc", 2: "def"},
        id_to_seq_id={"abc": 1, "def": 2},
    )


def test_safe_unpickler_blocks_exploit():
    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    buf = io.BytesIO()
    pickle.dump(Exploit(), buf)
    buf.seek(0)

    with pytest.raises(pickle.UnpicklingError, match="Forbidden"):
        SafeUnpickler(buf).load()


def test_safe_unpickler_loads_valid_data():
    buf = io.BytesIO()
    pickle.dump(_persistent_data(128), buf, pickle.HIGHEST_PROTOCOL)
    buf.seek(0)

    loaded = SafeUnpickler(buf).load()

    assert loaded.dimensionality == 128
    assert loaded.total_elements_added == 10
    assert loaded.id_to_label == {"abc": 1, "def": 2}
    assert loaded.label_to_id == {1: "abc", 2: "def"}
    assert loaded.id_to_seq_id == {"abc": 1, "def": 2}


def test_load_from_file_uses_safe_unpickler(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(_persistent_data(128), f, pickle.HIGHEST_PROTOCOL)

    loaded = PersistentData.load_from_file(str(path))

    assert loaded.dimensionality == 128
    assert loaded.id_to_label == {"abc": 1, "def": 2}


def test_load_from_file_uses_safe_unpickler_for_legacy_metadata(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(_persistent_data(None), f, pickle.HIGHEST_PROTOCOL)

    loaded = PersistentData.load_from_file(str(path), expected_dimensionality=7)

    assert loaded.dimensionality == 7
    assert loaded.id_to_label == {"abc": 1, "def": 2}


def test_load_from_file_blocks_malicious_pickle(tmp_path):
    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump(Exploit(), f, pickle.HIGHEST_PROTOCOL)

    with pytest.raises(pickle.UnpicklingError, match="Forbidden"):
        PersistentData.load_from_file(str(path))


def test_load_from_file_rejects_unexpected_root_object(tmp_path):
    path = tmp_path / "index_metadata.pickle"
    with path.open("wb") as f:
        pickle.dump({"dimensionality": 3}, f, pickle.HIGHEST_PROTOCOL)

    with pytest.raises(
        pickle.UnpicklingError, match="did not deserialize to PersistentData"
    ):
        PersistentData.load_from_file(str(path))
