package com.vladih.computer_vision.flutter_vision.models;

import android.content.Context;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class Yolov8 extends Yolo {
    public Yolov8(Context context,
                  String model_path,
                  boolean is_assets,
                  int num_threads,
                  boolean use_gpu,
                  String label_path,
                  int rotation) {
        super(context, model_path, is_assets, num_threads, use_gpu, label_path, rotation);
    }

    @Override
    protected List<float[]> filter_box(float[][][] model_outputs, float iou_threshold,
                                       float conf_threshold, float class_threshold, float input_width, float input_height) {
        try {
            //reshape [1,box+class,detected_box] to reshape [1,detected_box,box+class]
            model_outputs = reshape(model_outputs);
            List<float[]> pre_box = new ArrayList<>();
            int class_index = 4;
            int dimension = model_outputs[0][0].length;
            int rows = model_outputs[0].length;
            float x1, y1, x2, y2;
            for (int i = 0; i < rows; i++) {
                //convert xywh to xyxy
                x1 = (model_outputs[0][i][0] - model_outputs[0][i][2] / 2f);
                y1 = (model_outputs[0][i][1] - model_outputs[0][i][3] / 2f);
                x2 = (model_outputs[0][i][0] + model_outputs[0][i][2] / 2f);
                y2 = (model_outputs[0][i][1] + model_outputs[0][i][3] / 2f);
                float max = 0;
                int y = 0;
                int last_label_index = labels.size() + class_index;
                for (int j = class_index; (j < dimension && j < last_label_index); j++) {
                    if (model_outputs[0][i][j] < class_threshold) continue;
                    if (max < model_outputs[0][i][j]) {
                        max = model_outputs[0][i][j];
                        y = j;
                    }
                }
                int mask_index = class_index + labels.size();
                int mask_index_max_weight = 0;
                if (dimension > mask_index) {
                    //calculate index of segmentation mask with max. weight
                    float max_mask_weight = 0;
                    for (int j = mask_index; j < dimension; j++) {
                        float current_value = model_outputs[0][i][j];
                        if (max_mask_weight < current_value) {
                            max_mask_weight = current_value;
                            mask_index_max_weight = j;
                        }
                    }
                }
                if (max > 0) {
                    float[] tmp = new float[7];
                    tmp[0] = x1;
                    tmp[1] = y1;
                    tmp[2] = x2;
                    tmp[3] = y2;
                    tmp[4] = model_outputs[0][i][y];
                    tmp[5] = (y - class_index) * 1f;
                    if (mask_index_max_weight - mask_index >= 0) {
                        tmp[6] = (mask_index_max_weight - mask_index) * 1f;
                    } else {
                        tmp[6] = -1;
                    }
                    pre_box.add(tmp);
                }
            }
            if (pre_box.isEmpty()) return new ArrayList<>();
            //for reverse orden, insteand of using .reversed method
            Comparator<float[]> compareValues = (v1, v2) -> Float.compare(v1[1], v2[1]);
            //Collections.sort(pre_box,compareValues.reversed());
            Collections.sort(pre_box, compareValues);
            return nms(pre_box, iou_threshold);
        } catch (Exception e) {
            throw e;
        }
    }

    @Override
    protected List<Map<String, Object>> out(List<float[]> yolo_result, Vector<String> labels) {
        try {
            List<Map<String, Object>> result = new ArrayList<>();
            for (float[] box : yolo_result) {
                Map<String, Object> output = new HashMap<>();
                output.put("box", new float[]{box[0], box[1], box[2], box[3], box[4]}); //x1,y1,x2,y2,conf_class
                output.put("tag", labels.get((int) box[5]));
                result.add(output);
            }
            return result;
        } catch (Exception e) {
            throw e;
        }
    }

}
