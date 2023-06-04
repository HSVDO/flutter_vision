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
        Log.i("printShapes", format("Shapes retrieved: %s.%n", shapes));
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

    public void appendOutput() {
        appendLog("Output size: " + this.output[0].length + ":" + this.output[0][0].length + ", third dimension:" + this.output[0][0][0]);
        for (int i = 0; i < this.output[0].length; i++) {
            appendLog(Arrays.deepToString(this.output[i]));
            for (int j = 0; j < this.output[i][0].length; j++) {
                appendLog("###" + Arrays.toString(this.output[i][j]));
            }
        }
    }

    public void appendLog(String text) {
        File logFile = new File(this.context.getExternalFilesDir("txt"), "flutter_vision_log.txt");
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

    public List<Map<String, Object>> detect_task(ByteBuffer byteBuffer,
                                                 int source_height,
                                                 int source_width,
                                                 float iou_threshold,
                                                 float conf_threshold, float class_threshold) throws Exception {
        try {
            Log.i("detect_task", "Running interpreter: Creating arrays...");
            int[] shape = this.interpreter.getInputTensor(0).shape();
//            int[] tempShape1 = this.interpreter.getOutputTensor(0).shape();
//            int[] tempShape2 = this.interpreter.getOutputTensor(1).shape();
//            float[][][][][] tempOut = new float[tempShape1[0]][tempShape1[1]][tempShape1[2]][tempShape2[0]][tempShape2[1]];
            Log.i("detect_task", "Running interpreter...");
            this.interpreter.run(byteBuffer, this.output);
            Log.i("detect_task", "Ran interpreter...");
            Log.i("detect_task", format("Output: %s", Arrays.deepToString(this.output)));
            appendLog(format("Output: %s", Arrays.deepToString(this.output)));
            List<float[]> boxes = filter_box(this.output, iou_threshold, conf_threshold, class_threshold, shape[1], shape[2]);
            boxes = restore_size(boxes, shape[1], shape[2], source_width, source_height);
            Log.i("detect_task", format("Boxes: %s", Arrays.deepToString(boxes.toArray())));
            appendLog(format("Boxes: %s", Arrays.deepToString(boxes.toArray())));
            appendOutput();
            return out(boxes, this.labels);
        } catch (Exception e) {
            throw e;
        } finally {
            byteBuffer.clear();
        }
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
