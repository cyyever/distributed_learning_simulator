import os
import pickle
import sqlite3

from cyy_torch_toolbox.dataset import sub_dataset
from cyy_torch_toolbox.dataset_collection import DatasetCollectionConfig
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class Practitioner:
    def __init__(self, practitioner_id: int):
        self.practitioner_id = practitioner_id
        self._datasets: dict = {}

    @property
    def datasets(self):
        return self._datasets

    def add_dataset_collection(self, name: str, indices: dict) -> None:
        self._datasets[name] = indices

    def create_trainer(self, config):
        trainer = config.create_trainer()
        for phase in MachineLearningPhase:
            trainer.dataset_collection.transform_dataset(
                phase,
                lambda dataset, _, __: sub_dataset(
                    dataset, self.datasets[trainer.dataset_collection.name][phase]
                ),
            )
        return trainer


class PersistentPractitioner(Practitioner):
    __cache_dir: str = os.path.join(
        os.path.expanduser("~"), ".cache", "distributed_learning"
    )

    @classmethod
    def connect_db(cls):
        cache_dir = os.getenv("practitioner_cache_dir", cls.__cache_dir)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        conn = sqlite3.connect(os.path.join(os.path.join(cache_dir, "practitioner.db")))
        cur = conn.cursor()
        cur.execute(
            "create table if not exists practitioner (id integer primary key autoincrement)"
        )
        cur.execute(
            "create table if not exists dataset (practitioner_id integer primary key, datasets blob)"
        )
        conn.commit()
        return conn

    @classmethod
    def create_practitioner(cls) -> int:
        practitioner_id = None
        with cls.connect_db() as conn:
            cur = conn.cursor()
            cur.execute("insert into practitioner values(NULL)")
            for row in cur.execute("select max(id) from practitioner"):
                practitioner_id = row[0]
        assert practitioner_id is not None
        return practitioner_id

    def __init__(self, practitioner_id: int | None = None):
        if practitioner_id is None:
            super().__init__(practitioner_id=self.create_practitioner())
        else:
            super().__init__(practitioner_id=practitioner_id)
            self._datasets = self.__get_datasets()

    def iid_sample_dataset(self, name: str, percentage: float) -> None:
        config = DatasetCollectionConfig(dataset_name=name)
        dc = config.create_dataset_collection()
        result: dict = {}
        for phase in MachineLearningPhase:
            dataset_util = dc.get_dataset_util(phase=phase)
            result[phase] = sum(dataset_util.iid_sample(percentage).values(), start=[])
        self.add_dataset_collection(name, result)

    def non_iid_sample_dataset(self, name: str, percents: list[float]) -> None:
        config = DatasetCollectionConfig(dataset_name=name)
        dc = config.create_dataset_collection()
        result: dict = {}
        for phase in MachineLearningPhase:
            dataset_util = dc.get_dataset_util(phase=phase)
            result[phase] = sum(
                dataset_util.sample_by_labels(percents).values(), start=[]
            )
        self.add_dataset_collection(name, result)

    def add_dataset_collection(self, *args, **kwargs):
        super().add_dataset_collection(*args, **kwargs)
        self.__store_datasets()

    def __get_datasets(self) -> dict:
        dataset_blob = None
        with self.connect_db() as conn:
            cur = conn.cursor()
            for row in cur.execute(
                "select datasets from dataset where practitioner_id=?",
                (self.practitioner_id,),
            ):
                dataset_blob = row[0]

        if dataset_blob is None:
            return {}
        return pickle.loads(dataset_blob)

    def __store_datasets(self) -> None:
        dataset_blob = pickle.dumps(self.datasets)
        with self.connect_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "insert or replace into dataset values(?,?)",
                (self.practitioner_id, dataset_blob),
            )
