package com.vladih.computer_vision.flutter_vision.models;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

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
    protected final boolean use_nnapi;
    protected final String label_path;
    protected final int rotation;

    private GpuDelegate gpuDelegate;

    private NnApiDelegate nnApiDelegate;

    public Yolo(Context context,
                String model_path,
                boolean is_assets,
                int num_threads,
                boolean use_gpu,
                boolean use_nnapi,
                String label_path,
                int rotation) {
        this.context = context;
        this.model_path = model_path;
        this.is_assets = is_assets;
        this.num_threads = num_threads;
        this.use_gpu = use_gpu;
        this.use_nnapi = use_nnapi;
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
                        gpuDelegate = new GpuDelegate(gpuOptions.setQuantizedModelsAllowed(true));
                        interpreterOptions.addDelegate(gpuDelegate);

                    }
                } else if (use_nnapi) {
                    if (compatibilityList.isDelegateSupportedOnThisDevice()) {
                        // Initialize interpreter with NNAPI delegate for Android Pie or above
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                            NnApiDelegate.Options nnapiOptions =
                                    new NnApiDelegate.Options();
                            nnApiDelegate = new NnApiDelegate(nnapiOptions);
                            interpreterOptions.addDelegate(nnApiDelegate);
                            interpreterOptions.setUseNNAPI(true);
                        }
                    }
                }
                //batch, width, height, channels
                this.interpreter = new Interpreter(buffer, interpreterOptions);
                this.interpreter.allocateTensors();
                this.labels = load_labels(asset_manager, label_path);
                int[] shape = interpreter.getOutputTensor(0).shape();
                printShapes();
                this.output = new float[shape[0]][shape[1]][shape[2]];
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

//                appendOutputsToLog(outputs);

                float[][][] processed_masks = processMask_ultralytics(masks[0], resized_boxes.toArray(new float[0][]), source_width, source_height, true);

                List<Map<String, Object>> out = out(resized_boxes, this.labels);
                for (int i = 0; i < out.size(); i++) {
                    out.get(i).put("mask", convert_2d_array_dart_compatible(processed_masks[i]));
                    out.get(i).put("outline", convert_2d_array_dart_compatible(convertToOutline(clone(processed_masks[i]), (float[]) out.get(i).get("box"))));
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

    private float[][] convertToOutline(float[][] segmentationMask, float[] box) {
        int height = segmentationMask.length;
        int width = segmentationMask[0].length;
        float[][] outlineMask = new float[height][width];

        // Iterate over each pixel in the segmentation mask
        for (int y = (int) box[1]; y < (int) box[3]; y++) {
            for (int x = (int) box[0]; x < (int) box[2]; x++) {
                // Check if the current pixel is part of the object
                if (segmentationMask[y][x] != 0) {
                    // Check if any neighboring pixel is not part of the object
                    if (isOutline(segmentationMask, x, y)) {
                        // Set the pixel in the outline mask to indicate the outline
                        outlineMask[y][x] = 0;
                    } else {
                        // Set the pixel in the outline mask to indicate the outline
                        outlineMask[y][x] = 1;
                    }
                }
            }
        }

        return outlineMask;
    }

    private boolean isOutline(float[][] segmentationMask, int x, int y) {
        int height = segmentationMask.length;
        int width = segmentationMask[0].length;

        // Define the 8 possible neighboring pixels
        int[][] neighbors = {
                {-1, -1}, {0, -1}, {1, -1},
                {-1, 0}, {1, 0},
                {-1, 1}, {0, 1}, {1, 1}
        };

        int numOfNeighbors = 0;

        // Check if any neighboring pixel is part of the object
        for (int[] neighbor : neighbors) {
            int nx = x + neighbor[0];
            int ny = y + neighbor[1];

            // Check if the neighboring pixel is within the image bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (segmentationMask[ny][nx] != 0) {
                    numOfNeighbors++;
                }
            }
        }

        return numOfNeighbors >= 8;
    }

    public float[][][] processMask_ultralytics(float[][][] protos, float[][] bboxes, int source_width, int source_height, boolean upsample) {
        //Log.i("process_masks", format("source_width: %s, source_height: %s", source_width, source_height));
        int num_boxes = bboxes.length;
        if (num_boxes == 0) {
            return new float[0][source_width][source_height];
        }
        float[][] mask_weights = new float[bboxes.length][];
        int mask_weights_index = 6;
        for (int i = 0; i < num_boxes; i++) {
            mask_weights[i] = new float[bboxes[i].length - mask_weights_index];
            System.arraycopy(bboxes[i], mask_weights_index, mask_weights[i], 0, mask_weights[i].length);
        }

        int mask_width = protos.length;
        int mask_height = protos[0].length;
        int num_masks = protos[0][0].length;

        float[][][] masks = new float[num_boxes][source_width][source_height];

        for (int box_index = 0; box_index < num_boxes; box_index++) {
            for (int height_index = 0; height_index < mask_height; height_index++) {
                for (int width_index = 0; width_index < mask_width; width_index++) {
                    float sum = 0.0f;
                    for (int mask_index = 0; mask_index < num_masks; mask_index++) {
                        sum += mask_weights[box_index][mask_index] * protos[height_index][width_index][mask_index];
                    }
                    masks[box_index][height_index][width_index] = sigmoid(sum);
                }
            }
        }

        //transpose width and height
//        for (int box_index = 0; box_index < num_boxes; box_index++) {
//            masks[box_index] = transpose(masks[box_index]);
//        }

        float[][] downsampled_bboxes = clone(bboxes);
        for (float[] bbox : downsampled_bboxes) {
            bbox[0] *= (double) mask_width / source_width;
            bbox[1] *= (double) mask_height / source_height;
            bbox[2] *= (double) mask_width / source_width;
            bbox[3] *= (double) mask_height / source_height;
        }

//        crop_mask(masks, downsampled_bboxes);

        if (upsample) {
            masks = upsampleMask_ultralytics(masks, source_height, source_width, mask_height, mask_width);
//            masks = bicubicInterpolate(masks, source_width, source_height);
        }

        for (int box_index = 0; box_index < num_boxes; box_index++) {
            int array_width = masks[box_index].length;
            int array_height = masks[box_index][0].length;
            for (int width_index = 0; width_index < array_width; width_index++) {
                for (int height_index = 0; height_index < array_height; height_index++) {
                    masks[box_index][width_index][height_index] = masks[box_index][width_index][height_index] > 0.5 ? 1 : 0;
                }
            }
        }
        return masks;
    }

    float[][] clone(float[][] arrayToClone) {
        float[][] clone = arrayToClone.clone();
        for (int i = 0; i < clone.length; i++) {
            clone[i] = clone[i].clone();
        }
        return clone;
    }

    private void crop_mask(float[][][] masks, float[][] bboxes) {
        int num_boxes = bboxes.length;
        int mask_width = masks[0].length;
        int mask_height = masks[0][0].length;
        //Log.i("crop_mask", format("Num boxes: %s, mask_width: %s, mask_height: %s", num_boxes, mask_width, mask_height));
        for (int box_index = 0; box_index < num_boxes; box_index++) {
            for (int width_index = 0; width_index < mask_width; width_index++) {
                for (int height_index = 0; height_index < mask_height; height_index++) {
                    if (width_index < bboxes[box_index][0]) {
                        masks[box_index][width_index][height_index] = 0;
                    } else if (height_index < bboxes[box_index][1]) {
                        masks[box_index][width_index][height_index] = 0;
                    } else if (width_index > bboxes[box_index][2]) {
                        masks[box_index][width_index][height_index] = 0;
                    } else if (height_index > bboxes[box_index][3]) {
                        masks[box_index][width_index][height_index] = 0;
                    }
                }
            }
        }
    }

    public float[][] transpose(float[][] mat) {
        //Log.i("Transpose", format("Transposing: x: %s, y: %s", mat.length, mat[0].length));
        float[][] result = new float[mat[0].length][mat.length];
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                result[j][i] = mat[i][j];
            }
        }
        return result;
    }

    private float[][][] bicubicInterpolate(float[][][] mask, int source_width, int source_height) {
        float[][][] upsampledMask = new float[mask.length][source_height][source_width];
        for (int i = 0; i < mask.length; i++) {
            upsampledMask[i] = bicubicInterpolate(mask[i].clone(), source_width, source_height);
        }
        return upsampledMask;
    }

    public float[][] bicubicInterpolate(float[][] input, int outputWidth, int outputHeight) {
        int inputWidth = input[0].length;
        int inputHeight = input.length;

        float[][] output = new float[outputHeight][outputWidth];

        float xRatio = (float) (inputWidth - 1) / outputWidth;
        float yRatio = (float) (inputHeight - 1) / outputHeight;

        for (int y = 0; y < outputHeight; y++) {
            float yScaled = y * yRatio;
            int yFloor = (int) Math.floor(yScaled);
            float yFraction = yScaled - yFloor;

            for (int x = 0; x < outputWidth; x++) {
                float xScaled = x * xRatio;
                int xFloor = (int) Math.floor(xScaled);
                float xFraction = xScaled - xFloor;

                float[] values = new float[16];

                for (int j = -1; j <= 2; j++) {
                    for (int i = -1; i <= 2; i++) {
                        int inputX = clamp(xFloor + i, 0, inputWidth - 1);
                        int inputY = clamp(yFloor + j, 0, inputHeight - 1);
                        values[(j + 1) * 4 + (i + 1)] = input[inputY][inputX];
                    }
                }

                float interpolatedValue = bicubicInterpolation(values, xFraction, yFraction);
                output[y][x] = interpolatedValue;
            }
        }

        return output;
    }

    private float bicubicInterpolation(float[] values, float x, float y) {
        float[][] coefficients = {
                {-0.5f, 1.5f, -1.5f, 0.5f},
                {1.0f, -2.5f, 2.0f, -0.5f},
                {-0.5f, 0.0f, 0.5f, 0.0f},
                {0.0f, 1.0f, 0.0f, 0.0f}
        };

        float result = 0.0f;

        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                float weight = coefficients[j][i] * bicubicKernel(x - i) * bicubicKernel(y - j);
                result += values[j * 4 + i] * weight;
            }
        }

        return result;
    }

    private float bicubicKernel(float x) {
        float a = -0.5f;
        float absX = Math.abs(x);

        if (absX <= 1.0f) {
            return (a + 2) * (absX * absX * absX) - (a + 3) * (absX * absX) + 1;
        } else if (absX < 2.0f) {
            return a * (absX * absX * absX) - 5 * a * (absX * absX) + 8 * a * absX - 4 * a;
        } else {
            return 0.0f;
        }
    }

    private int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }


    private float[][][] upsampleMask_ultralytics(float[][][] mask, int h, int w, int mh, int mw) {
        float[][][] upsampledMask = new float[mask.length][h][w];
        for (int i = 0; i < mask.length; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) {
                    int sj = (int) Math.floor((double) j * (mh - 1) / (h - 1));
                    int sk = (int) Math.floor((double) k * (mw - 1) / (w - 1));
                    upsampledMask[i][j][k] = mask[i][sj][sk];
                }
            }
        }
        return upsampledMask;
    }

    public float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
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
            if (interpreter != null) {
                interpreter.close();
            }
            if (gpuDelegate != null) {
                gpuDelegate.close();
            }
            if (nnApiDelegate != null) {
                nnApiDelegate.close();
            }
        } catch (Exception e) {
            throw e;
        }
    }

}
