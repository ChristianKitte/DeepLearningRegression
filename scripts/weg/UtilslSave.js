/**
 * Gibt True zurück, wenn es sich bei dem übergebenen Parameter um ein Array mit einer
 * Länge von größer 0 handelt, es also einen Inhalt hat / nicht leer ist
 * @param arr Das zu prüfende Array
 * @returns {boolean} True, wenn es ein nicht leeres Array ist
 */
function isArrayAndFilled(arr) {
    return (Array.isArray(arr) && arr.length > 0);
}

/**
 * Mischt ein Array mit dem Fisher-Yates Algorithmus durch und erzeugt aus dem übergebenen
 * Array ein tf.Dataset
 * @param arr Das zugrunde liegende Array
 * @returns {*} Ein auf das Array basierendes tf.Dataset
 */
function getShuffledDataSetByArray(arr) {
    tf.util.shuffle(arr);
    return tf.data.array(arr);
}

/**
 * Mischt ein Array mit dem Fisher-Yates Algorithmus durch und separiert die Feature und Label
 * Eigenschaft aus dem übergebenen Arrays in zwei separate Arrays
 * @param arr Ein Array mit Elementen (JSON) mit den gesuchten Werten
 * @returns {{features: *[], labels: *[]}} Features und Labels in zwei getrennten Arrays
 */
function getShuffledFeatureAndLabelColumn(arr) {
    let newFeatureArray = [];
    let newLabelArray = [];

    tf.util.shuffle(arr);

    arr.forEach(function (item) {
        newFeatureArray.push(item.feature);
        newLabelArray.push(item.label);
    })

    return {
        features: newFeatureArray,
        labels: newLabelArray
    };
}

/**
 * Normalisiert den übergebenen Tensor mit min-max-scaling - inputTensor.sub(min).div(max.sub(min)) - und
 * gibt den normierten Tensor plus das Minimum und Maximum zurück
 * @param inputTensor Der zu normalisierende Tensor
 * @param inputTensor
 * @returns {{normierterTensor: *, min: *, max: *}}
 */
function normalizeTensor(inputTensor) {
    return tf.tidy(() => {
        const max = inputTensor.max();
        const min = inputTensor.min();
        const normalizedTensor = inputTensor.sub(min).div(max.sub(min));

        return {
            normierterTensor: normalizedTensor,
            min: max,
            max: min
        };
    });
}

/**
 * Denormalisiert den übergebenen Tensor mit min-max-scaling
 * @param inputTensor Der zu denormalisierende Tensor
 * @param min Der bei der Normalisierung genutzte Minwert
 * @param max Der bei der Normalisierung genutzte Maxwert
 * @returns {*} Ein neuer Tensor mit den ursprünglichen Werten
 */
function unNormalizeTensor(inputTensor, min, max) {
    return tf.tidy(() => {
        return inputTensor.mul(max.sub(min)).add(min);
    });
}

/***
 * Druckt eine Übersicht des gegebenen TensorFlow Models in dem durch ID bezeichneten DIV
 * Container aus.
 * @param divContainerId Die ID eines DIV Containers
 * @param model Ein Tensorflow Model
 */
function printModel(divContainerId, model) {
    tfvis.show.modelSummary(document.getElementById(divContainerId), model);
}

/***
 * Druckt eine Beschreibung des gegebenen TensorFlow Models in dem durch ID bezeichneten DIV
 * Container aus.
 * @param divContainerId Die ID eines DIV Containers
 * @param modelDescription Ein Tensorflow Model
 */
function printModelDescription(divContainerId, modelDescription) {
    tfvis.show.modelSummary(document.getElementById(divContainerId), modelDescription.toJSON());
}

/**
 * Liest die Optionsgruppe mit dem übergebenen ID aus und gibt den selektierten Index
 * zurück
 * @param id Die ID einer Optionsgruppe
 * @returns {number} Den Index des ausgewählten Indexes oder -1;
 */
function getRadioOption(id) {
    const radios = document.getElementsByName(id);
    let index = -1;

    for (let radio of radios) {
        if (radio.checked) {
            index = +radio.value;
        }
    }

    return index;
}