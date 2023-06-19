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
    protected static String old_model_path;
    protected final boolean is_assets;
    protected final int num_threads;
    protected final boolean use_gpu;
    protected final String label_path;
    protected final int rotation;

    private boolean isClosed;

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
            if (this.interpreter == null || !this.model_path.equals(old_model_path)) {
                old_model_path = this.model_path;
                if(interpreter != null){
                    close();
                }
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
                isClosed = false;
                Log.i("initialize_model", "Marked model as NOT closed.");
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
                List<float[][]> converted_masks = crop_dimensions(masks);

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

    private List<List<float[]>> processMaskOutput(List<float[]> restored_boxes, List<float[][]> converted_masks, int maskWidth, int maskHeight, int source_width, int source_height) {
        List<List<float[]>> post_processed_masks = new ArrayList<>();
        int dimension_x = converted_masks.get(0).length;
        int dimension_y = converted_masks.get(0)[0].length;

        //TODO sigmoid on masks: https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation/blob/main/yoloseg/YOLOSeg.py#L100

        float ratioWidth = (((float) source_width) / ((float) maskWidth));
        float ratioHeight = (((float) source_height) / ((float) maskHeight));

        List<float[]> downscaled_boxes = downscale_boxes(restored_boxes, ratioWidth, ratioHeight);

        //Get mask weights per class
        List<float[]> box_mask_weights = new ArrayList<>();

        for (float[] box : restored_boxes) {
            int mask_index_in_box = 6; //0 = x1, 1=y1, 2=x2, 3=y2, 4=class conf, 5=class, 5-37 = mask weights
            float[] box_mask = new float[box.length - mask_index_in_box];
            System.arraycopy(box, mask_index_in_box, box_mask, 0, box.length - mask_index_in_box);
            box_mask_weights.add(box_mask);
        } //returns List<array with 32 weights>

        //multiply each mask by its corresponding weight
        List<float[][]> mask_per_box = mask_per_box_old(converted_masks, box_mask_weights);

        //upscale masks
        for (float[][] mask : mask_per_box) {
            Log.i("Interpolating", format("Before: %s", Arrays.deepToString(mask)));
            float[][] interpolated = bicubicInterpolate(mask, ratioWidth, ratioHeight);
//            round_to_zero_or_one(interpolated);
            Log.i("Interpolating", format("After: %s", Arrays.deepToString(interpolated)));

            List<float[]> final_mask = convert_2d_array_dart_compatible(interpolated);

            post_processed_masks.add(final_mask);
        }

        return post_processed_masks;
    }

    private static void round_to_zero_or_one(float[][] interpolated) {
        for (int x = 0; x < interpolated.length; x++) {
            for (int y = 0; y < interpolated[0].length; y++) {
                interpolated[x][y] = (interpolated[x][y] > 0.5) ? 1 : 0;
            }
        }
    }

    public List<float[][]> mask_per_box_new(List<float[][]> mask_outputs, List<float[]> box_mask_weights) {
        List<float[][]> mask_per_box = new ArrayList<>();
        for (float[] mask_predictions : box_mask_weights) {
            for (float[][] mask_output : mask_outputs) {
                int num_mask = mask_predictions.length;
                int reshapeColumns = mask_output.length * mask_output[0].length;
                float[][] reshapedMaskOutput = new float[num_mask][reshapeColumns];
                float[][] masks = new float[num_mask][mask_output.length * mask_output[0].length];
                for (int i = 0; i < num_mask; i++) {
                    for (int j = 0; j < mask_output.length; j++) {
                        System.arraycopy(mask_output[j], 0, reshapedMaskOutput[i], j * mask_output[0].length, mask_output[0].length);
                    }
                }
                for (int i = 0; i < num_mask; i++) {
                    for (int j = 0; j < reshapedMaskOutput[0].length; j++) {
                        float dotProduct = 0.0f;
                        for (int k = 0; k < mask_predictions.length; k++) {
                            dotProduct += mask_predictions[k] * reshapedMaskOutput[i][k];
                        }
                        masks[i][j] = sigmoid(dotProduct);
                    }
                }
                mask_per_box.add(masks);
            }
        }

        return mask_per_box;
    }

    public List<float[][]> mask_per_box_old(List<float[][]> sigmoid, List<float[]> box_mask_weights) {
        List<float[][]> mask_per_box = new ArrayList<>();
        for (float[] box : box_mask_weights) {
            float[][] summed_mask_for_this_box = null;
            for (int mask_weight_index = 0; mask_weight_index < sigmoid.size(); mask_weight_index++) {
                float[][] mask = sigmoid.get(mask_weight_index).clone();
                mask = transpose(mask);
//                mask = sigmoid(mask);
                for (int x_index = 0; x_index < sigmoid.size(); x_index++) {
                    for (int y_index = 0; y_index < sigmoid.get(0).length; y_index++) {
                        mask[x_index][y_index] = box[mask_weight_index] * mask[x_index][y_index];
                    }
                }
//                mask = sigmoid(mask);
                mask = transpose(mask);
                //here is the mask multiplied and ready to be summed up
                if (summed_mask_for_this_box == null) {
                    summed_mask_for_this_box = mask.clone();
                } else {
                    for (int x_index = 0; x_index < sigmoid.size(); x_index++) {
                        for (int y_index = 0; y_index < sigmoid.get(0).length; y_index++) {
                            summed_mask_for_this_box[x_index][y_index] += mask[x_index][y_index];
                        }
                    }
                }
            }
            mask_per_box.add(summed_mask_for_this_box);
            break;
        }
        return mask_per_box;
    }

    /**
     * Transposes a matrix.
     * Assumption: mat is a non-empty matrix. i.e.:
     * 1. mat != null
     * 2. mat.length > 0
     * 3. For every i, mat[i].length are equal and mat[i].length > 0
     */
    public float[][] transpose(float[][] mat) {
        float[][] result = new float[mat[0].length][mat.length];
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                result[j][i] = mat[i][j];
            }
        }
        return result;
    }

    private List<float[][]> sigmoid(List<float[][]> converted_masks) {
        List<float[][]> masks = new ArrayList<>();
        for (float[][] mask : converted_masks) {
            masks.add(sigmoid(mask));
        }
        return masks;
    }

    public float[][] sigmoid(float[][] array) {
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                array[i][j] = sigmoid(array[i][j]);
            }
        }
        return array;
    }

    public float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
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

    private List<float[][]> crop_dimensions(float[][][][] masks) {
        List<float[][]> converted_masks = new ArrayList<>();
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
            converted_masks.add(mask);
        }
        return converted_masks;
    }

    public List<float[]> convert_2d_array_dart_compatible(float[][] array) {
        //convert array float[][] to List<float[]>. See dart method channel data type limitations for further information
        return new ArrayList<>(Arrays.asList(array));
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
            isClosed = true;
            if (interpreter != null)
                interpreter.close();
        } catch (Exception e) {
            throw e;
        }
    }

    public boolean isClosed() {
        return isClosed;
    }
}
