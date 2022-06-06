/**
 * Das aktuell gültige DataSet
 * @type {*[]}
 */
let currentDataSet = [];

/**
 * Erzeugt auf Basis des aktuellen DataSet den feature und label Tensor. Zudem eine normalisierte Version
 * mit den genutzten Min und Max Werten.
 * @returns {*}
 */
function getData() {
    return tf.tidy(() => {
        // Durchmischen, trennen des DataSets
        const shuffled = getShuffledFeatureAndLabelColumn(currentDataSet);
        const featureTensor = tf.tensor(shuffled.features);
        const labelTensor = tf.tensor(shuffled.labels);

        // Normalisieren von Features und Labels
        const normalisierungFeatures = normalizeTensor(featureTensor);
        const normalizedFeatures = normalisierungFeatures.normierterTensor;
        const minFeature = normalisierungFeatures.min;
        const maxFeature = normalisierungFeatures.max;

        const normalisierungLabels = normalizeTensor(labelTensor);
        const normalizedLabels = normalisierungLabels.normierterTensor;
        const minLabels = normalisierungLabels.min;
        const maxLabels = normalisierungLabels.max;

        return workDataSet = {
            featureTensor: featureTensor,
            labelTensor: labelTensor,
            //normalisierungFeatures: normalisierungFeatures,
            normalizedFeatures: normalizedFeatures,
            minFeature: minFeature,
            maxFeature: maxFeature,
            //normalisierungLabels: normalisierungLabels,
            normalizedLabels: normalizedLabels,
            minLabels: minLabels,
            maxLabels: maxLabels,
        };
    });
}

// Helper functions used for sampling
function getUniformDistributedRandomNumber(min, max) {
    return Math.random() * (max - min) + min;
}

function addGaussianNoise(label, mean, variance) {
    const labelNoisy = label * Math.sqrt(variance) + mean;
    return labelNoisy;
}

function calcSum(dataArray) {
    let total = 0;

    for (let i = 0; i < dataArray.length; i++) {
        total += dataArray[i].label;
    }

    return total;
}

function calcMean(dataArray) {
    let sum = calcSum(dataArray);
    return sum / dataArray.length;
}

function getDataArrayWithNoise(dataArray, noise) {
    const mean = calcMean(dataArray);

    for (let item of dataArray) {
        let newLabel = addGaussianNoise(item.label, mean, noise);
        item.label = newLabel;
    }

    return dataArray;
}

function getDataArray(countNumbers) {
    let dataArray = [];

    for (let i = 0; i < countNumbers; i++) {
        let feature = getUniformDistributedRandomNumber(-1, 1);
        let label = (feature + 0.8) * (feature - 0.2) * (feature - 0.3) * (feature - 0.6);

        dataArray.push({
            feature: feature,
            label: label,
        });
    }

    return dataArray;
}

