//package com.vladih.computer_vision.flutter_vision.models;
//
//import org.opencv.core.Core;
//import org.opencv.core.CvType;
//import org.opencv.core.Mat;
//import org.opencv.core.Rect;
//import org.opencv.core.Scalar;
//import org.opencv.core.Size;
//import org.opencv.imgproc.Imgproc;
//
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//import java.util.Random;
//
//public class YoloSegmentation {
//
//    private float confThreshold;
//    private float iouThreshold;
//    private int numMasks;
//    private onnxruntime.InferenceSession session;
//    private List<String> classNames;
//    private double[][] colors;
//
//    public YoloSegmentation(String path, float confThreshold, float iouThreshold, int numMasks) {
//        this.confThreshold = confThreshold;
//        this.iouThreshold = iouThreshold;
//        this.numMasks = numMasks;
//        this.classNames = new ArrayList<>(Arrays.asList("person"));
//
//        // Create a list of colors for each class where each color is a tuple of 3 integer values
//        Random rng = new Random(3);
//        this.colors = new double[this.classNames.size()][3];
//        for (int i = 0; i < this.classNames.size(); i++) {
//            for (int j = 0; j < 3; j++) {
//                this.colors[i][j] = rng.nextDouble() * 255;
//            }
//        }
//
//        // Initialize model
//        initializeModel(path);
//    }
//
//    public List<Rect> segmentObjects(Mat image) {
//        Mat inputTensor = prepareInput(image);
//
//        // Perform inference on the image
//        List<Mat> outputs = inference(inputTensor);
//
//        Mat boxOutput = outputs.get(0);
//        Mat maskOutput = outputs.get(1);
//
//        List<Rect> boxes = processBoxOutput(boxOutput);
//        List<Mat> maskMaps = processMaskOutput(maskOutput, outputs.get(2));
//
//        return boxes;
//    }
//
//    private void initializeModel(String path) {
//        try {
//            this.session = new onnxruntime.InferenceSession(path);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//
//    private Mat prepareInput(Mat image) {
//        int imgHeight = image.rows();
//        int imgWidth = image.cols();
//
//        Mat inputImg = new Mat();
//        Imgproc.cvtColor(image, inputImg, Imgproc.COLOR_BGR2RGB);
//
//        // Resize input image
//        Size size = new Size(this.inputWidth, this.inputHeight);
//        Imgproc.resize(inputImg, inputImg, size);
//
//        // Scale input pixel values to 0 to 1
//        inputImg.convertTo(inputImg, CvType.CV_32F, 1.0 / 255.0);
//
//        return inputImg;
//    }
//
//    private List<Mat> processMaskOutput(Mat maskOutput, Mat maskPrediction) {
//        List<Mat> maskMaps = new ArrayList<>();
//        float[][] maskOutputData = new float[maskOutput.rows()][maskOutput.cols()];
//
//        if (maskOutputData.length == 0) {
//            return maskMaps;
//        }
//
//        float[] maskPredictionData = new float[(int) maskPrediction.total()];
//        maskPrediction.get(0, 0, maskPredictionData);
//
//        // Calculate the mask maps for each box
//        int numMask = maskOutput.rows();
//        int maskHeight = maskOutput.cols();
//        int maskWidth = maskOutput.cols();
//
//        float[][] sigmoidInput = reshape(maskPredictionData, numMask, -1);
//        float[][] sigmoidResult = sigmoid(multiply(maskPredictionData, sigmoidInput));
//        sigmoidResult = reshape(sigmoidResult, -1, maskHeight, maskWidth);
//
//        // Downscale the boxes to match the mask size
//        List<Rect> scaleBoxes = rescaleBoxes(boxes, new Size(this.imgWidth, this.imgHeight), new Size(maskWidth, maskHeight));
//
//        int ratioWidth = (int) (this.imgWidth / maskWidth);
//        int ratioHeight = (int) (this.imgHeight / maskHeight);
//        Size blurSize = new Size(ratioWidth, ratioHeight);
//
//        for (int i = 0; i < scaleBoxes.size(); i++) {
//            Rect scaleBox = scaleBoxes.get(i);
//            int scaleX1 = (int) Math.floor(scaleBox.x);
//            int scaleY1 = (int) Math.floor(scaleBox.y);
//            int scaleX2 = (int) Math.ceil(scaleBox.x + scaleBox.width);
//            int scaleY2 = (int) Math.ceil(scaleBox.y + scaleBox.height);
//
//            int x1 = (int) Math.floor(boxes[i].x);
//            int y1 = (int) Math.floor(boxes[i].y);
//            int x2 = (int) Math.ceil(boxes[i].x + boxes[i].width);
//            int y2 = (int) Math.ceil(boxes[i].y + boxes[i].height);
//
//            Mat scaleCropMask = new Mat(maskOutput.submat(scaleY1, scaleY2, scaleX1, scaleX2));
//            Imgproc.resize(scaleCropMask, scaleCropMask, new Size(x2 - x1, y2 - y1));
//
//            Imgproc.blur(scaleCropMask, scaleCropMask, blurSize);
//
//            Mat cropMask = new Mat();
//            Core.compare(scaleCropMask, new Scalar(0.5), cropMask, Core.CMP_GT);
//            cropMask.convertTo(cropMask, CvType.CV_8U);
//
//            Mat maskMap = new Mat(new Size(this.imgWidth, this.imgHeight), CvType.CV_8U, new Scalar(0));
//            cropMask.copyTo(maskMap.submat(y1, y2, x1, x2));
//
//            maskMaps.add(maskMap);
//        }
//
//        return maskMaps;
//    }
//
//
//    private List<Rect> rescaleBoxes(List<Rect> boxes, Size inputShape, Size imageShape) {
//        List<Rect> rescaledBoxes = new ArrayList<>();
//        double[] inputShapeArr = {inputShape.width, inputShape.height, inputShape.width, inputShape.height};
//        double[] imageShapeArr = {imageShape.width, imageShape.height, imageShape.width, imageShape.height};
//
//        for (Rect box : boxes) {
//            double[] boxArr = {box.x, box.y, box.width, box.height};
//            double[] rescaledBoxArr = divide(boxArr, inputShapeArr);
//            rescaledBoxArr = multiply(rescaledBoxArr, imageShapeArr);
//
//            rescaledBoxes.add(new Rect((int) rescaledBoxArr[0], (int) rescaledBoxArr[1],
//                    (int) rescaledBoxArr[2], (int) rescaledBoxArr[3]));
//        }
//
//        return rescaledBoxes;
//    }
//
//    private float[][] reshape(float[] arr, int rows, int cols) {
//        float[][] reshapedArr = new float[rows][cols];
//        int k = 0;
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < cols; j++) {
//                reshapedArr[i][j] = arr[k++];
//            }
//        }
//        return reshapedArr;
//    }
//
//    private float[][] sigmoid(float[][] arr) {
//        float[][] result = new float[arr.length][arr[0].length];
//        for (int i = 0; i < arr.length; i++) {
//            for (int j = 0; j < arr[0].length; j++) {
//                result[i][j] = (float) (1 / (1 + Math.exp(-arr[i][j])));
//            }
//        }
//        return result;
//    }
//
//    private float[][] multiply(float[][] arr1, double[] arr2) {
//        float[][] result = new float[arr1.length][arr1[0].length];
//        for (int i = 0; i < arr1.length; i++) {
//            for (int j = 0; j < arr1[0].length; j++) {
//                result[i][j] = (float) (arr1[i][j] * arr2[j]);
//            }
//        }
//        return result;
//    }
//
//    private double[] divide(double[] arr1, double[] arr2) {
//        double[] result = new double[arr1.length];
//        for (int i = 0; i < arr1.length; i++) {
//            result[i] = arr1[i] / arr2[i];
//        }
//        return result;
//    }
//
//    private void getInputDetails() {
//        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
//        options.addExecutionProvider(CPUExecutionProviderFactory.INSTANCE);
//        try (OrtSession session = OrtEnvironment.getEnvironment().createSession(modelPath, options)) {
//            this.inputNames = session.getInputNames();
//            this.outputNames = session.getOutputNames();
//
//            OrtTensor inputTensor = (OrtTensor) session.getInputTypeInfo(0).createTensor();
//            int[] inputShape = inputTensor.getInfo().getShape();
//            this.inputHeight = inputShape[2];
//            this.inputWidth = inputShape[3];
//        } catch (OrtException e) {
//            e.printStackTrace();
//        }
//    }
//
//    private void getOutputDetails() {
//        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
//        options.addExecutionProvider(CPUExecutionProviderFactory.INSTANCE);
//        try (OrtSession session = OrtEnvironment.getEnvironment().createSession(modelPath, options)) {
//            this.outputNames = session.getOutputNames();
//        } catch (OrtException e) {
//            e.printStackTrace();
//        }
//    }
//
//
//}
