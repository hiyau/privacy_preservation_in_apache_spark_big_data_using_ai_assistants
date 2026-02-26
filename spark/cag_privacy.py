import os
os.environ["HADOOP_HOME"] = "C:/hadoop"
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("CAGPrivacy")
    .config("spark.sql.warehouse.dir", "file:/C:/temp/spark-warehouse")
    .getOrCreate()
)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sha2, lit
import os




from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

spark = SparkSession.builder \
    .appName("Healthcare Privacy Aggregation") \
    .getOrCreate()

df = spark.read.csv(
    "data/raw/privacy_preserving_healthcare_10000_records.csv",
    header=True,
    inferSchema=True
)

# =========================
# REMOVE ALL IDENTIFIERS
# =========================
safe_df = df.drop(
    "patient_id",
    "name",
    "insurance_id",
    "email",
    "phone",
    "address"
)

# =========================
# AGGREGATION ONLY
# =========================
agg_df = (
    safe_df
    .groupBy("disease")
    .agg(count("*").alias("patient_count"))
)

# Save ONLY aggregated data
agg_df.coalesce(1).write.mode("overwrite").csv(
    "data/processed/healthcare_cag",
    header=True
)

spark.stop()
