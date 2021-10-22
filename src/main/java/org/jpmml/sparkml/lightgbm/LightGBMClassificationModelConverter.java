package org.jpmml.sparkml.lightgbm;

import com.microsoft.ml.spark.lightgbm.LightGBMBooster;
import com.microsoft.ml.spark.lightgbm.LightGBMClassificationModel;
import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ClassificationModelConverter;
import org.jpmml.sparkml.model.HasTreeOptions;

public class LightGBMClassificationModelConverter extends ClassificationModelConverter<LightGBMClassificationModel>
    implements HasTreeOptions {
  public LightGBMClassificationModelConverter(LightGBMClassificationModel model) {
    super(model);
  }
  @Override
  public Model encodeModel(Schema schema) {
    LightGBMClassificationModel model = getTransformer();
    LightGBMBooster booster = model.getModel();
    return BoosterUtil.encodeBinaryClassificationBooster(booster, schema);
  }
}
