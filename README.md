JPMML-SparkML-LightGBM
=====================
JPMML-SparkML plugin for converting [LightGBM-Spark](https://github.com/Azure/mmlspark/blob/master/docs/lightgbm.md) models to PMML.

# Prerequisites #
* [Apache Spark](http://spark.apache.org/) 2.4.x
* [LightGBM-Spark](https://github.com/Azure/mmlspark) 0.18.1.

# Installation #

Enter the project root directory and build using [Apache Maven](http://maven.apache.org/):
```
mvn clean install
```

The build installs JPMML-SparkML-LightGBM library into local repository using coordinates `org.jpmml:jpmml-sparkml-lightgbm:1.0-SNAPSHOT`.

# Usage #

The JPMML-SparkML-LightGBM library extends the [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library with support for `com.microsoft.ml.spark.lightgbm.LightGBMClassificationModel` prediction model classes.


add  `org.jpmml:jpmml-sparkml-lightgbm:1.0-SNAPSHOT.jar` to CLASSPATH
```python

import  mmlspark
import mmlspark.train
from pyspark.ml import PipelineModel

df = spark.sql("select * from algo_dc_ml_split_data")
model = PipelineModel.load("/user/turing/lightgbm_spark")

from pyspark2pmml import PMMLBuilder

pmmlBuilder = PMMLBuilder(spark.sparkContext, df, model)

pmmlBuilder.buildFile("algo_dc_ml_2c_lightgbm_spark.xml")

hdfs_client.upload("algo_dc_ml_2c_lightgbm_spark.xml","${hdfs_path}")

```

# License #

JPMML-SparkML-LightGBM is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Other licenses are available on request.

