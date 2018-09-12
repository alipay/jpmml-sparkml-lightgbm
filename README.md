JPMML-SparkML-LightGBM
=====================
JPMML-SparkML plugin for converting [LightGBM-Spark](https://github.com/Azure/mmlspark/blob/master/docs/lightgbm.md) models to PMML.

# Prerequisites #
* [Apache Spark](http://spark.apache.org/) 2.2.X or 2.3.X.
* [LightGBM-Spark](https://github.com/Azure/mmlspark) 0.13.dev17+1.g4a9bd92.

# Installation #

Enter the project root directory and build using [Apache Maven](http://maven.apache.org/):
```
mvn clean install
```

The build installs JPMML-SparkML-LightGBM library into local repository using coordinates `org.jpmml:jpmml-sparkml-lightgbm:1.0-SNAPSHOT`.

# Usage #

The JPMML-SparkML-LightGBM library extends the [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library with support for `com.microsoft.ml.spark.LightGBMClassificationModel` prediction model classes.

[Add an example here]

# License #

JPMML-SparkML-LightGBM is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Other licenses are available on request.

