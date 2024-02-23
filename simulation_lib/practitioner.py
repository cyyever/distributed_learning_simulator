from cyy_torch_toolbox import Config, MachineLearningPhase, Trainer
from cyy_torch_toolbox.dataset import SamplerBase, SplitBase


class Practitioner:
    def __init__(self, practitioner_id: int) -> None:
        self.__id: int = practitioner_id
        self.__worker_id = practitioner_id
        self._dataset_sampler: dict[str, SamplerBase | SplitBase] = {}

    @property
    def id(self):
        return self.__id

    @property
    def worker_id(self):
        return self.__worker_id

    def set_worker_id(self, worker_id: int) -> None:
        self.__worker_id = worker_id

    def set_sampler(self, sampler: SamplerBase | SplitBase) -> None:
        collection_name = sampler.dataset_collection.name
        assert collection_name not in self._dataset_sampler
        self._dataset_sampler[collection_name] = sampler

    def has_dataset(self, name: str) -> bool:
        return name in self._dataset_sampler

    def create_trainer(self, config: Config) -> Trainer:
        sampler = self._dataset_sampler[config.dc_config.dataset_name]
        assert sampler.dataset_collection is not None
        dc = sampler.dataset_collection
        if isinstance(sampler, SplitBase):
            dc = sampler.sample(part_id=self.__worker_id)
        else:
            dc = sampler.sample()
        trainer = config.create_trainer(dc=dc)
        trainer.dataset_collection.remove_dataset(phase=MachineLearningPhase.Test)
        assert sampler.dataset_collection.has_dataset(MachineLearningPhase.Test)
        return trainer


# class PersistentPractitioner(Practitioner):
#     __cache_dir: str = os.path.join(
#         os.path.expanduser("~"), ".cache", "distributed_learning"
#     )

#     def __init__(self, practitioner_id: int) -> None:
#         super().__init__(practitioner_id=practitioner_id)
#         for name, sampler in self.__get_datasets().items():
#             super().set_sampler(name, sampler)

#     @classmethod
#     def connect_db(cls):
#         cache_dir = os.getenv("practitioner_cache_dir", cls.__cache_dir)
#         if not os.path.isdir(cache_dir):
#             os.makedirs(cache_dir, exist_ok=True)
#         conn = sqlite3.connect(os.path.join(os.path.join(cache_dir, "practitioner.db")))
#         cur = conn.cursor()
#         cur.execute(
#             "create table if not exists practitioner (id integer primary key autoincrement)"
#         )
#         cur.execute(
#             "create table if not exists dataset (practitioner_id integer primary key, datasets blob)"
#         )
#         conn.commit()
#         return conn

#     @classmethod
#     def create_practitioner(cls) -> int:
#         practitioner_id: None | int = None
#         with cls.connect_db() as conn:
#             cur = conn.cursor()
#             cur.execute("insert into practitioner values(NULL)")
#             for row in cur.execute("select max(id) from practitioner"):
#                 practitioner_id = row[0]
#         assert practitioner_id is not None
#         return practitioner_id

#     def set_sampler(self, *args, **kwargs):
#         super().set_sampler(*args, **kwargs)
#         self.__store_datasets()

#     def __get_datasets(self) -> dict:
#         dataset_blob = None
#         with self.connect_db() as conn:
#             cur = conn.cursor()
#             for row in cur.execute(
#                 "select datasets from dataset where practitioner_id=?",
#                 (self.practitioner_id,),
#             ):
#                 dataset_blob = row[0]

#         if dataset_blob is None:
#             return {}
#         return pickle.loads(dataset_blob)

#     def __store_datasets(self) -> None:
#         dataset_blob = pickle.dumps(self._dataset_sampler)
#         with self.connect_db() as conn:
#             cur = conn.cursor()
#             cur.execute(
#                 "insert or replace into dataset values(?,?)",
#                 (self.practitioner_id, dataset_blob),
#             )
