import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="distributed_learning_simulator",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/distributed_learning_simulator",
    package_data={"": ["conf/*/*.yaml", "conf/*.yaml"]},
    include_package_data=True,
    package_dir={
        "distributed_learning_simulator": ".",
        "distributed_learning_simulator.algorithm": "algorithm",
        "distributed_learning_simulator.conf": "conf",
        "distributed_learning_simulator.server": "server",
        "distributed_learning_simulator.topology": "topology",
        "distributed_learning_simulator.worker": "worker",
        "distributed_learning_simulator.sampler": "sampler",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
