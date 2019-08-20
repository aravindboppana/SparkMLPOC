package com.clairvoyant.insight.bigdata

import org.apache.spark.sql.SparkSession

object SparkConfigFactory {

  def createSparkSession(master: String,appName: String): SparkSession = {

    val spark = SparkSession
      .builder
      .master(master)
      .appName(appName)
      .getOrCreate()

    spark
  }

}
