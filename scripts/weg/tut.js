/**
 * Constants for model presets
 */

const UNDER_FITTING = {
    model: {
        hiddenLayers: 2,
        neuronsPerLayer: 6,
        activationFunction: "relu",
    },
    training: {
        epochs: 75,
        optimizer: "adam",
    },
};

const OVER_FITTING = {
    model: {
        hiddenLayers: 12,
        neuronsPerLayer: 32,
        activationFunction: "relu",
    },
    training: {
        epochs: 200,
        optimizer: "adam",
    },
};

const BEST_FIT = {
    model: {
        hiddenLayers: 6,
        neuronsPerLayer: 32,
        activationFunction: "relu",
    },
    training: {
        epochs: 100,
        optimizer: "adam",
    },
};

/**
 * Sampling of training and test data
 */

const N = [5, 10, 20, 50, 100];
var sampled = [];
var data = [];

// Helper functions used for sampling
function getUniformDistributedRandomNumber(min, max) {
    return Math.random() * (max - min) + min;
}

function addGaussianNoise(y, mean, variance) {
    const yNoisy = y * Math.sqrt(variance) + mean;
    return yNoisy;
}

function calcSum(array) {
    let total = 0;
    for (let i = 0; i < array.length; i++) {
        total += array[i].y;
    }
    return total;
}

function calcMean(array) {
    let sum = calcSum(array);
    return sum / array.length;
}

// Generate x and y values for n of N
for (const n of N) {
    const tmpArr = [];
    for (let i = 0; i < n; i++) {
        let x = getUniformDistributedRandomNumber(-1, 1);
        let y = (x + 0.8) * (x - 0.2) * (x - 0.3) * (x - 0.6);

        tmpArr.push({
            x: x,
            y: y,
        });
    }
    sampled.push(tmpArr);
}

// Add gaussian to noise to y(x)
for (const array of sampled) {
    const mean = calcMean(array);
    for (const element of array) {
        element.y = addGaussianNoise(element.y, mean, 0.3);
    }
}

/**
 * Handle user interaction
 */

function handleSamplesRadioChange() {
    const radios = document.getElementsByName("inlineRadioOptions");
    let index;
    for (let radio of radios) {
        if (radio.checked) {
            index = +radio.value;
        }
    }
    data = sampled[index];
    renderData(data);
}

function handlePresetsUnderFitting() {
    document.getElementById("hiddenLayers").value =
        UNDER_FITTING.model.hiddenLayers;
    document.getElementById("neuronsPerLayer").value =
        UNDER_FITTING.model.neuronsPerLayer;
    document.getElementById("epochs").value = UNDER_FITTING.training.epochs;
    hiddenLayers = UNDER_FITTING.model.hiddenLayers;
    neuronsPerLayer = UNDER_FITTING.model.neuronsPerLayer;
    epochs = UNDER_FITTING.training.epochs;
}

function handlePresetsOverFitting() {
    document.getElementById("hiddenLayers").value =
        OVER_FITTING.model.hiddenLayers;
    document.getElementById("neuronsPerLayer").value =
        OVER_FITTING.model.neuronsPerLayer;
    document.getElementById("epochs").value = OVER_FITTING.training.epochs;
    hiddenLayers = OVER_FITTING.model.hiddenLayers;
    neuronsPerLayer = OVER_FITTING.model.neuronsPerLayer;
    epochs = OVER_FITTING.training.epochs;
}

function handlePresetsBestFit() {
    document.getElementById("hiddenLayers").value = BEST_FIT.model.hiddenLayers;
    document.getElementById("neuronsPerLayer").value =
        BEST_FIT.model.neuronsPerLayer;
    document.getElementById("epochs").value = BEST_FIT.training.epochs;
    hiddenLayers = BEST_FIT.model.hiddenLayers;
    neuronsPerLayer = BEST_FIT.model.neuronsPerLayer;
    epochs = BEST_FIT.training.epochs;
}

function handleHiddenLayersChange() {
    hiddenLayers = +document.getElementById("hiddenLayers").value;
}

function handleNeuronsChange() {
    neuronsPerLayer = +document.getElementById("neuronsPerLayer").value;
    console.log(neuronsPerLayer)
}

function handleBatchSizeChange() {
    batchSize = +document.getElementById("batchSize").value;
}

function handleEpochsChange() {
    epochs = +document.getElementById("epochs").value;
}

/**
 * Render visualization
 */

function renderData(data) {
    const values = data;
    tfvis.render.scatterplot(
        { name: "x/x(y)" },
        { values },
        {
            xLabel: "x",
            yLabel: "y(x)",
            height: 300,
        }
    );
}

/**
 * Creation, training and testing of model
 */

// Model parameters
var hiddenLayers;
var neuronsPerLayer;
var activationFunction;

// Training parameters
var batchSize = 32;
var epochs = 100;
var optimizer;
var learningRate;

var model;

function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    for (let i = 0; i < hiddenLayers; i++) {
        model.add(tf.layers.dense({ units: neuronsPerLayer, activation: "relu" }));
    }

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    /*   model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));

    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" })); */

    return model;
}

function create() {
    model = createModel();
    tfvis.show.modelSummary({ name: "Model Summary" }, model);
}

function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() =>{
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map((d) => d.x);
        const labels = data.map((d) => d.y);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor
            .sub(inputMin)
            .div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor
            .sub(labelMin)
            .div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        };
    });
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ["mse"],
    });

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: "Training Performance" },
            ["loss", "mse"],
            { height: 200, callbacks: ["onEpochEnd"] }
        ),
    });
}

var tensorData;

async function train() {
    tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;
    await trainModel(model, inputs, labels);
}

function testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));
        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] };
    });

    const originalPoints = inputData.map((d) => ({
        x: d.x,
        y: d.y,
    }));

    tfvis.render.scatterplot(
        { name: "Model Predictions vs Original Data" },
        {
            values: [originalPoints, predictedPoints],
            series: ["original", "predicted"],
        },
        {
            xLabel: "x",
            yLabel: "y",
            height: 300,
        }
    );
}

function test() {
    testModel(model, data, tensorData);
}

document.addEventListener("DOMContentLoaded", tfvis.visor());

const tooltipTriggerList = document.querySelectorAll(
    '[data-bs-toggle="tooltip"]'
);
const tooltipList = [...tooltipTriggerList].map(
    (tooltipTriggerEl) => new bootstrap.Tooltip(tooltipTriggerEl)
);