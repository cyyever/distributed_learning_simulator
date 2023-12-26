import os

import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

package_dir = {
    "distributed_learning_simulator": "./simulation_lib",
    "distributed_learning_simulator.conf": "./conf",
}

for dirname in os.listdir("./simulation_lib"):
    assert dirname != "fed_css"
    if os.path.isdir(os.path.join("./simulation_lib", dirname)):
        package_dir[
            f"distributed_learning_simulator.{dirname}"
        ] = f"./simulation_lib/{dirname}"

for dirname in os.listdir("./simulation_lib/method"):
    if os.path.isdir(os.path.join("./simulation_lib/method", dirname)):
        package_dir[
            f"distributed_learning_simulator.method.{dirname}"
        ] = f"./simulation_lib/method/{dirname}"


setuptools.setup(
    name="distributed_learning_simulator",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/distributed_learning_simulator",
    package_dir=package_dir,
    package_data={
        "distributed_learning_simulator.conf": ["*/*.yaml", "*.yaml"]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
