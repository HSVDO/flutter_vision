package com.vladih.computer_vision.flutter_vision.models;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import androidx.annotation.NonNull;

import static com.vladih.computer_vision.flutter_vision.FlutterVisionPlugin.yolo_typing;
import static java.lang.Math.min;
import static java.lang.String.format;

public class Yolo {
    protected float[][][] output;
    protected Interpreter interpreter;
    protected Vector<String> labels;
    protected final Context context;
    protected final String model_path;
    protected final boolean is_assets;
    protected final int num_threads;
    protected final boolean use_gpu;
    protected final String label_path;
    protected final int rotation;

    public Yolo(Context context,
                String model_path,
                boolean is_assets,
                int num_threads,
                boolean use_gpu,
                String label_path,
                int rotation) {
        this.context = context;
        this.model_path = model_path;
        this.is_assets = is_assets;
        this.num_threads = num_threads;
        this.use_gpu = use_gpu;
        this.label_path = label_path;
        this.rotation = rotation;
        appendLog(model_path);
    }

    //    public Vector<String> getLabels(){return this.labels;}
    public Tensor getInputTensor() {
        return this.interpreter.getInputTensor(0);
    }

    public void initialize_model() throws Exception {
        AssetManager asset_manager = null;
        MappedByteBuffer buffer = null;
        FileChannel file_channel = null;
        FileInputStream input_stream = null;
        try {
            if (this.interpreter == null) {
                if (is_assets) {
                    asset_manager = context.getAssets();
                    AssetFileDescriptor file_descriptor = asset_manager.openFd(
                            this.model_path);
                    input_stream = new FileInputStream(file_descriptor.getFileDescriptor());

                    file_channel = input_stream.getChannel();
                    buffer = file_channel.map(
                            FileChannel.MapMode.READ_ONLY, file_descriptor.getStartOffset(),
                            file_descriptor.getLength()
                    );
                    file_descriptor.close();

                } else {
                    input_stream = new FileInputStream(new File(this.model_path));
                    file_channel = input_stream.getChannel();
                    buffer = file_channel.map(FileChannel.MapMode.READ_ONLY, 0, file_channel.size());

                }
                CompatibilityList compatibilityList = new CompatibilityList();
                Interpreter.Options interpreterOptions = new Interpreter.Options();
                interpreterOptions.setNumThreads(num_threads);
                if (use_gpu) {
                    if (compatibilityList.isDelegateSupportedOnThisDevice()) {
                        GpuDelegate.Options gpuOptions = compatibilityList.getBestOptionsForThisDevice();
                        interpreterOptions.addDelegate(
                                new GpuDelegate(gpuOptions.setQuantizedModelsAllowed(true)));
                    }
                }
                //batch, width, height, channels
                this.interpreter = new Interpreter(buffer, interpreterOptions);
                this.interpreter.allocateTensors();
                this.labels = load_labels(asset_manager, label_path);
                int[] shape = interpreter.getOutputTensor(0).shape();
                printShapes();
                this.output = new float[shape[0]][shape[1]][shape[2]];
            }
        } catch (Exception e) {
            throw e;
        } finally {

            if (buffer != null)
                buffer.clear();
            if (file_channel != null)
                if (file_channel.isOpen())
                    file_channel.close();
            if (file_channel != null)
                if (file_channel.isOpen())
                    input_stream.close();
        }
    }

    private void printShapes() {
        List<String> shapes = new ArrayList<>();
        for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
            shapes.add(format("Input tensor #%s has shape: %s", i, Arrays.toString(interpreter.getInputTensor(i).shape())));
        }
        for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
            shapes.add(format("Output tensor #%s has shape: %s", i, Arrays.toString(interpreter.getOutputTensor(i).shape())));
        }
        appendLog(format("Shapes retrieved: %s.%n", shapes));
        Log.i("initialize_model", format("Model shapes retrieved: %s.%n", shapes));
    }

    protected Vector<String> load_labels(AssetManager asset_manager, String label_path) throws Exception {
        BufferedReader br = null;
        try {
            if (asset_manager != null) {
                br = new BufferedReader(new InputStreamReader(asset_manager.open(label_path)));
            } else {
                br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(label_path))));
            }
            String line;
            Vector<String> labels = new Vector<>();
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            return labels;
        } catch (Exception e) {
            throw new Exception(e.getMessage());
        } finally {
            if (br != null) {
                br.close();
            }
        }
    }

    public void appendLog(String text) {
        //TODO remove this method/mark as comment
        if (!"img".equals(yolo_typing) || 1 == 1) {
            return;
        }
        String fileName = this.model_path.substring(this.model_path.lastIndexOf("/") + 1).replace(".", "");
        File logFile = new File(this.context.getExternalFilesDir("txt"), "fv_log" + fileName + ".txt");
        if (!logFile.exists()) {
            try {
                logFile.createNewFile();
            } catch (IOException e) {
                Log.e("appendLog", "Cannot create file:", e);
            }
        }
        try {
            //BufferedWriter for performance, true to set append to file flag
            BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
            buf.append(text);
            buf.newLine();
            buf.close();
        } catch (IOException e) {
            Log.e("appendLog", "Cannot write to file:", e);
        }
    }

    //https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_segmenter
    public List<Map<String, Object>> detect_task(ByteBuffer byteBuffer,
                                                 int source_height,
                                                 int source_width,
                                                 float iou_threshold,
                                                 float conf_threshold, float class_threshold) throws Exception {

        try {
            if (hasMultipleOutput()) {
                Map<Integer, Object> outputs = new HashMap<>();
                for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
                    int[] shape = interpreter.getOutputTensor(i).shape();
                    outputs.put(i, Array.newInstance(float.class, shape));
                }
                Object[] inputs = {byteBuffer};
                this.interpreter.runForMultipleInputsOutputs(inputs, outputs);
                int[] shape = interpreter.getInputTensor(0).shape();
                float[][][] boxArray = (float[][][]) outputs.get(0);

                List<float[]> boxes = filter_box(boxArray, iou_threshold, conf_threshold, class_threshold, shape[1], shape[2]);
                List<float[]> resized_boxes = restore_size(boxes, shape[1], shape[2], source_width, source_height);

                int boxCount = 1;
                for (float[] box : resized_boxes) {
                    appendLog("Box#" + boxCount++ + ": " + Arrays.toString(box));
                }

                float[][][][] masks = (float[][][][]) outputs.get(1);
                List<List<float[]>> converted_masks = convert_masks_dart_compatible(masks);

                appendOutputsToLog(outputs);

                shape = interpreter.getOutputTensor(1).shape();
                List<List<float[]>> post_processed_masks = processMaskOutput(resized_boxes, converted_masks, shape[1], shape[2], source_width, source_height);

                List<Map<String, Object>> out = out(resized_boxes, this.labels);
                for (int i = 0; i < out.size(); i++) {
                    out.get(i).put("mask", post_processed_masks.get(i));
                }
                return out;
            }
            int[] shape = this.interpreter.getInputTensor(0).shape();
            this.interpreter.run(byteBuffer, this.output);
            List<float[]> boxes = filter_box(this.output, iou_threshold, conf_threshold, class_threshold, shape[1], shape[2]);
            boxes = restore_size(boxes, shape[1], shape[2], source_width, source_height);


            int boxCount = 1;
            for (float[] box : boxes) {
                appendLog("Box#" + boxCount++ + ": " + Arrays.toString(box));
            }

            return out(boxes, this.labels);
        } catch (Exception e) {
            throw e;
        } finally {
            byteBuffer.clear();
        }
    }

    private List<List<float[]>> processMaskOutput(List<float[]> restored_boxes, List<List<float[]>> converted_masks, int maskWidth, int maskHeight, int source_width, int source_height) {
        List<List<float[]>> post_processed_masks = new ArrayList<>();

        //TODO sigmoid on masks: https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation/blob/main/yoloseg/YOLOSeg.py#L100
        List<List<float[]>> sigmoid = converted_masks;

        Log.i("processMaskOutput", format("Calculating scale ratio: sourceW=%s, maskW=%s", source_width, maskWidth));
        float ratioWidth = (((float) source_width) / ((float) maskWidth));
        float ratioHeight = (((float) source_height) / ((float) maskHeight));

        List<float[]> downscaled_boxes = downscale_boxes(restored_boxes, ratioWidth, ratioHeight);

        List<List<float[]>> rescaled_masks = new ArrayList<>();

        //Get mask weights per class
        List<float[]> box_mask_weights = new ArrayList<>();

        for (float[] box : restored_boxes) {
            int mask_index_in_box = 6; //0 = x1, 1=y1, 2=x2, 3=y2, 4=class conf, 5=class, 5-37 = mask weights
            float[] box_mask = new float[box.length - mask_index_in_box];
            System.arraycopy(box, mask_index_in_box, box_mask, 0, box.length - mask_index_in_box);
            box_mask_weights.add(box_mask);
        } //returns List<array with 32 weights>

        //multiply each mask by its corresponding weight
        List<List<float[]>> mask_per_box = new ArrayList<>();
        for (float[] box : box_mask_weights) {
            List<float[]> mask_for_this_box = new ArrayList<>();
            for (int mask_weight_index = 0; mask_weight_index < sigmoid.size(); mask_weight_index++) {
                List<float[]> mask = sigmoid.get(mask_weight_index);
                List<float[]> mask_in_processing = new ArrayList<>();
                for (int x_index = 0; x_index < mask.size(); x_index++) {
                    float[] column = new float[mask.get(x_index).length];
                    for (int y_index = 0; y_index < column.length; y_index++) {
                        column[y_index] = box[mask_weight_index] * mask.get(x_index)[y_index];
                    }
                    mask_in_processing.add(column);
                }
                //here is the mask multiplied and ready to be summed up
                if (mask_for_this_box.isEmpty()) {
                    mask_for_this_box = mask_in_processing;
                } else {
                    for (int i = 0; i < mask_for_this_box.size(); i++) {
                        float[] column = mask_for_this_box.get(i);
                        for (int j = 0; j < column.length; j++) {
                            float[] column_to_add = mask_in_processing.get(i);
                            column[j] += column_to_add[j];
                        }
                    }
                }
            }
            mask_per_box.add(mask_for_this_box);
        }

        //upscale masks
        for (List<float[]> mask : mask_per_box) {

            int numRows = mask.size();
            int numCols = mask.get(0).length;

            float[][] mask_as_array = new float[numRows][numCols];

            for (int i = 0; i < numRows; i++) {
                float[] row = mask.get(i);
                System.arraycopy(row, 0, mask_as_array[i], 0, numCols);
            }


            List<float[]> final_mask = new ArrayList<>(Arrays.asList(bicubicInterpolate(mask_as_array, ratioWidth, ratioHeight)));

            Log.i("upscaling_mask", format("final mask is of size %s", final_mask.size()));

            Log.i("upscaling_mask", format("final mask has format (%s, %s)", final_mask.size(), final_mask.get(0).length));
            post_processed_masks.add(final_mask);
        }

        return post_processed_masks;
    }

    public float bicubicInterpolation(float[] p, float x) {
        return p[1] + 0.5f * x * (p[2] - p[0] +
                x * (2.0f * p[0] - 5.0f * p[1] + 4.0f * p[2] - p[3] +
                        x * (3.0f * (p[1] - p[2]) + p[3] - p[0])));
    }

    public float[][] bicubicInterpolate(float[][] data, float scaleX, float scaleY) {
        int height = data.length;
        int width = data[0].length;

        int newHeight = (int) (height * scaleY);
        int newWidth = (int) (width * scaleX);

        Log.i("bicubic_interpolate", format("Upscaling by scales %s&%s from %s, %s to %s, %s",
                scaleX, scaleY, width, height, newWidth, newHeight));

        float[][] interpolatedData = new float[newHeight][newWidth];

        for (int y = 0; y < newHeight; y++) {
            float yScaled = y / scaleY;

            for (int x = 0; x < newWidth; x++) {
                float xScaled = x / scaleX;

                int xFloor = (int) Math.floor(xScaled);
                int yFloor = (int) Math.floor(yScaled);

                float[] p = new float[4];

                for (int i = -1; i < 3; i++) {
                    int yPos = Math.min(Math.max(yFloor + i, 0), height - 1);
                    p[i + 1] = bicubicInterpolation(data[yPos], xScaled - xFloor);
                }

                interpolatedData[y][x] = bicubicInterpolation(p, yScaled - yFloor);
            }
        }

        Log.i("bicubic_interpolate", format("Array to be returned hast size %s, %s", interpolatedData.length, interpolatedData[0].length));
        return interpolatedData;
    }


    private List<float[]> downscale_boxes(List<float[]> restored_boxes, float ratioWidth, float ratioHeight) {
        List<float[]> result = new ArrayList<>();
        for (float[] array : restored_boxes) {
            result.add(new float[]{
                    array[0] / ratioWidth, //x1
                    array[1] / ratioHeight, //y1
                    array[2] / ratioWidth, //x2
                    array[3] / ratioHeight, //y2
                    array[4], //conf
                    array[5] //label
            });
        }
        return result;
    }

    @NonNull
    private static Map<String, Object> build_mask_map(List<List<float[]>> converted_masks) {
        Map<String, Object> masks_map = new HashMap<>();
        if (!converted_masks.isEmpty()) {
            masks_map.put("masks", converted_masks);
        }
        return masks_map;
    }

    private List<List<float[]>> convert_masks_dart_compatible(float[][][][] masks) {
        List<List<float[]>> converted_masks = new ArrayList<>();
        if (masks == null) {
            return converted_masks;
        }
        for (int mask_index = 0; mask_index < masks[0][0][0].length; mask_index++) {
            float[][] mask = new float[masks[0].length][masks[0][0].length];
            for (int i = 0; i < mask.length; i++) {
                for (int j = 0; j < mask[0].length; j++) {
                    float mask_value = masks[0][i][j][mask_index];
                    mask[i][j] = mask_value;
                }
            }
            //convert array float[][] to List<float[]>. See dart method channel data type limitations for further information
            List<float[]> converted_mask = new ArrayList<>(Arrays.asList(mask));
            if (!converted_mask.isEmpty()) {
                converted_masks.add(converted_mask);
            }
        }
        return converted_masks;
    }

    private void appendOutputsToLog(Map<Integer, Object> outputs) {
        appendLog("Retrieved " + outputs.size() + " outputs.");
        appendLog("#0=" + Arrays.deepToString((float[][][]) outputs.get(0)));
        appendLog("#1=" + Arrays.deepToString((float[][][][]) outputs.get(1)));
    }

    private boolean hasMultipleOutput() {
        return this.interpreter.getOutputTensorCount() > 1;
    }

    protected List<float[]> filter_box(float[][][] model_outputs, float iou_threshold,
                                       float conf_threshold, float class_threshold, float input_width, float input_height) {
        try {
            List<float[]> pre_box = new ArrayList<>();
            int conf_index = 4;
            int class_index = 5;
            int dimension = model_outputs[0][0].length;
            int rows = model_outputs[0].length;
            float x1, y1, x2, y2, conf;
            for (int i = 0; i < rows; i++) {
                //convert xywh to xyxy
                x1 = (model_outputs[0][i][0] - model_outputs[0][i][2] / 2f) * input_width;
                y1 = (model_outputs[0][i][1] - model_outputs[0][i][3] / 2f) * input_height;
                x2 = (model_outputs[0][i][0] + model_outputs[0][i][2] / 2f) * input_width;
                y2 = (model_outputs[0][i][1] + model_outputs[0][i][3] / 2f) * input_height;
                conf = model_outputs[0][i][conf_index];
                if (conf < conf_threshold) continue;
                float max = 0;
                int y = 0;
                for (int j = class_index; j < dimension; j++) {
                    if (model_outputs[0][i][j] < class_threshold) continue;
                    if (max < model_outputs[0][i][j]) {
                        max = model_outputs[0][i][j];
                        y = j;
                    }
                }
                if (max > 0) {
                    float[] tmp = new float[6];
                    tmp[0] = x1;
                    tmp[1] = y1;
                    tmp[2] = x2;
                    tmp[3] = y2;
                    tmp[4] = model_outputs[0][i][y];
                    tmp[5] = (y - class_index) * 1f;
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

    protected static List<float[]> nms(List<float[]> boxes, float iou_threshold) {
        try {
            for (int i = 0; i < boxes.size(); i++) {
                float[] box = boxes.get(i);
                for (int j = i + 1; j < boxes.size(); j++) {
                    float[] next_box = boxes.get(j);
                    float x1 = Math.max(next_box[0], box[0]);
                    float y1 = Math.max(next_box[1], box[1]);
                    float x2 = Math.min(next_box[2], box[2]);
                    float y2 = Math.min(next_box[3], box[3]);

                    float width = Math.max(0, x2 - x1);
                    float height = Math.max(0, y2 - y1);

                    float intersection = width * height;
                    float union = (next_box[2] - next_box[0]) * (next_box[3] - next_box[1])
                            + (box[2] - box[0]) * (box[3] - box[1]) - intersection;
                    float iou = intersection / union;
                    if (iou > iou_threshold) {
                        boxes.remove(j);
                        j--;
                    }
                }
            }
            return boxes;
        } catch (Exception e) {
            Log.e("nms", e.getMessage());
            throw e;
        }
    }

    protected List<float[]> restore_size(List<float[]> nms,
                                         int input_width,
                                         int input_height,
                                         int src_width,
                                         int src_height) {
        try {
            //restore size after scaling, larger images
            if (src_width > input_width || src_height > input_height) {
                float gainx = src_width / (float) input_width;
                float gainy = src_height / (float) input_height;
                for (int i = 0; i < nms.size(); i++) {
                    nms.get(i)[0] = min(src_width, Math.max(nms.get(i)[0] * gainx, 0));
                    nms.get(i)[1] = min(src_height, Math.max(nms.get(i)[1] * gainy, 0));
                    nms.get(i)[2] = min(src_width, Math.max(nms.get(i)[2] * gainx, 0));
                    nms.get(i)[3] = min(src_height, Math.max(nms.get(i)[3] * gainy, 0));
                }
                //restore size after padding, smaller images
            } else {
                float padx = (src_width - input_width) / 2f;
                float pady = (src_height - input_height) / 2f;
                for (int i = 0; i < nms.size(); i++) {
                    nms.get(i)[0] = min(src_width, Math.max(nms.get(i)[0] + padx, 0));
                    nms.get(i)[1] = min(src_height, Math.max(nms.get(i)[1] + pady, 0));
                    nms.get(i)[2] = min(src_width, Math.max(nms.get(i)[2] + padx, 0));
                    nms.get(i)[3] = min(src_height, Math.max(nms.get(i)[3] + pady, 0));
                }
            }
            return nms;
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    protected static float[][][] reshape(float[][][] input) {
        final int x = input.length;
        final int y = input[0].length;
        final int z = input[0][0].length;
        // Convert input array to OpenCV Mat [x y z] to [x z y]
//        Mat inputMat = new Mat(x*y, z, CvType.CV_32F);
//        for (int i = 0; i < x; i++) {
//            for (int j = 0; j < y; j++) {
//                inputMat.put(y*i+j, 0, input[i][j]);
//            }
//        }
//        // Reshape Mat
//        Mat outputMat = inputMat.reshape(y,x*z);
//
//        // Convert output Mat to float[][][] array
//        float[][][] output = new float[x][z][y];
//        for (int i = 0; i < x; i++) {
//            for (int j = 0; j < z; j++) {
//                outputMat.row(z*i+j).get(0, 0, output[i][j]);
//            }
//        }
//        // Convert output Mat to float[][][] array
        float[][][] output = new float[x][z][y];
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < z; k++) {
                    output[i][k][j] = input[i][j][k];
                }
            }
        }
        return output;
    }

    protected List<Map<String, Object>> out(List<float[]> yolo_result, Vector<String> labels) {
        try {
            List<Map<String, Object>> result = new ArrayList<>();
            //utils.getScreenshotBmp(bitmap, "current");
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

    public void close() {
        try {
            if (interpreter != null)
                interpreter.close();
        } catch (Exception e) {
            throw e;
        }
    }
}
