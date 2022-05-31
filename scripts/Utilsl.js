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
 * Normalisiert den übergebenen Tensor mit min-max-scaling
 * @param inputTensor Der zu normalisierende Tensor
 * @returns {*} Ein neuer normalisierter Tensor
 */
function normalizeTensor(inputTensor) {
    const max = inputTensor.max();
    const min = inputTensor.min();

    return inputTensor.sub(min).div(max.sub(min));
}