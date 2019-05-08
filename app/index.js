import * as tf from "@tensorflow/tfjs";
import { ObjectDetectionImageSynthesizer } from "./synthetic_images";

const canvas = document.getElementById("data-canvas");
const status = document.getElementById("status");
const testModel = document.getElementById("test");
const loadHostedModel = document.getElementById("load-hosted-model");
const inferenceTimeMs = document.getElementById("inference-time-ms");
const trueObjectClass = document.getElementById("true-object-class");
const predictedObjectClass = document.getElementById("predicted-object-class");

let idTimeInterval = null;

const LOCAL_MODEL_PATH = "mobilenet.json";
const HOSTED_MODEL_PATH =
  "https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json";

const updateUI = (isRectangle, predit) => {
  const trueClassName = isRectangle ? "rectangle" : "triangle";
  trueObjectClass.textContent = trueClassName;

  const shapeClassificationThreshold = canvas.width / 2;
  const predictClassName =
    predit > shapeClassificationThreshold ? "rectangle" : "triangle";
  predictedObjectClass.textContent = predictClassName;

  if (predictClassName === trueClassName) {
    predictedObjectClass.classList.remove("shape-class-wrong");
    predictedObjectClass.classList.add("shape-class-correct");
  } else {
    predictedObjectClass.classList.remove("shape-class-correct");
    predictedObjectClass.classList.add("shape-class-wrong");
    clearInterval(idTimeInterval);
    testModel.disabled = true;
  }
};

const generateTensorsImgTarg = (fullCanvas, boundingBox) =>
  tf.tidy(() => {
    const imagesTensor = tf.browser.fromPixels(fullCanvas);
    const targetTensor = tf.tensor1d(boundingBox);
    console.log(tf.stack([imagesTensor]));
    return {
      images: tf.stack([imagesTensor]),
      targets: tf.stack([targetTensor])
    };
  });

const arrAvg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

// const t0 = tf.util.now();
// inferenceTimeMs.textContent = `${(tf.util.now() - t0).toFixed(2)}`;
const runAndVisualizeInference = async model => {
  // Sintetice una imagen de entrada y muéstrela en el lienzo.
  const synth = new ObjectDetectionImageSynthesizer(canvas);

  const { fullCanvas, boundingBox } = await synth.generateExample();

  const { images, targets } = generateTensorsImgTarg(fullCanvas, boundingBox);

// Ejecuta inferencia con el modelo.
const modelOut2 = await model.predict(images).data();

  // Visualice los cuadros de delimitación verdaderos y predichos.
  const targetsArray = Array.from(await targets.data());
  console.log(targetsArray);
  const percentage = (arrAvg(targetsArray) / arrAvg(modelOut2.slice(1))) * 100;
  synth.drawBoundingBoxes(targetsArray, modelOut2.slice(1), percentage);

  // Mostrar el verdadero y predecir las clases de objetos.
  // updateUI(isRectangle, modelOut[0]);

  // Tensor memory cleanup.
  // tf.dispose([images, targets]);
};

const init = async () => {
  const loadModal = async modelPath => {
    try {
        status.textContent = `Loading model ...`;
        const model = await tf.loadLayersModel(HOSTED_MODEL_PATH);
        model.summary();
        console.log(model)
        testModel.disabled = false;
        status.textContent = `Loaded model successfully. Now click "Test Model".`;
        runAndVisualizeInference(model);

      idTimeInterval = setInterval(() => {
        runAndVisualizeInference(model);
      }, 3000);
      testModel.addEventListener("click", () =>
        runAndVisualizeInference(model)
      );
    } catch (err) {
      status.textContent = `Failed to load model`;
    }
  };
  loadModal(HOSTED_MODEL_PATH);
  // loadModal(LOCAL_MODEL_PATH);
};

init();

// const catElement = document.getElementById("cat");
// const img = tf.browser.fromPixels(catElement).toFloat();
// const offset = tf.scalar(127.5);
// const normalized = img.sub(offset).div(offset);
// const batched = normalized.reshape([1, 224, 224, 3]);
// const modelOut2 = await model.predict(batched).data();
// console.log(modelOut2);
