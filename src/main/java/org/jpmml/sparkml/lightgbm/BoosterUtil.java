package org.jpmml.sparkml.lightgbm;

import com.microsoft.ml.spark.lightgbm.LightGBMBooster;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.OpType;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.*;
import org.jpmml.converter.Schema;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.lightgbm.GBDT;
import org.jpmml.lightgbm.HasLightGBMOptions;
import org.jpmml.lightgbm.LightGBMUtil;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

public class BoosterUtil {

  private BoosterUtil() {}

  public static MiningModel encodeBinaryClassificationBooster(LightGBMBooster booster, Schema schema) {
    byte[] bytes = booster.model().getBytes();
    GBDT gbdt = null;
    try (InputStream is = new ByteArrayInputStream(bytes)) {
      gbdt = LightGBMUtil.loadGBDT(is);
    } catch (IOException ioe) {
      throw new RuntimeException(ioe);
    }

    Function<Feature, Feature> function = feature -> {
      if (feature instanceof BinaryFeature) {
        return (BinaryFeature)feature;
      } else {
        return feature.toContinuousFeature(DataType.FLOAT);
      }
    };

    Schema lgbmSchema = schema.toTransformedSchema(function);
    Map<String, Object> options = new LinkedHashMap<>();
    options.put(HasLightGBMOptions.OPTION_COMPACT, true);
    options.put(HasLightGBMOptions.OPTION_NUM_ITERATION, null);
    MiningModel model = gbdt.encodeMiningModel(options, lgbmSchema)
        .setOutput(ModelUtil.createPredictedOutput(FieldName.create("gbtValue"), OpType.CONTINUOUS, DataType.DOUBLE));
    return MiningModelUtil.createBinaryLogisticClassification(
        model, 2d, 0d,
        RegressionModel.NormalizationMethod.LOGIT, false, lgbmSchema);
  }
}
