/**
 * Das aktuell gültige DataSet
 * @type {*[]}
 */
let currentDataSet = [];

/**
 * Das DropDown Element zur Auswahl des DataSets
 * @type {HTMLElement}
 */
let dataSetDropDow = document.getElementById("dataset");

/**
 * Handelt das Event OnChange des Dropdown für die Auswahl des DataSets
 * @param event
 */
dataSetDropDow.onchange = function (event) {
    let infoString = document.getElementById("on-load-string");
    let countDataArray;

    if (this.value === "1") { // 5
        countDataArray = 5;
        infoString.textContent = "DataSet 05 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "2") { // 10
        countDataArray = 10;
        infoString.textContent = "DataSet 10 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "3") { // 20
        countDataArray = 20;
        infoString.textContent = "DataSet 20 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "4") { // 50
        countDataArray = 50;
        infoString.textContent = "DataSet 50 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "5") { // 100
        countDataArray = 100;
        infoString.textContent = "DataSet 100 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    }

    let dataArray = getDataArray(countDataArray);
    let dataArrayNoisy = getDataArrayWithNoise(dataArray, 0.3);

    currentDataSet = dataArrayNoisy;

    DrawGraph("dataSetGraphFunction", "Funktionswerte der Rohdaten", dataArrayNoisy);
};

/**
 * Erzeugt auf Basis des aktuellen DataSet den feature und label Tensor. Zudem eine normalisierte Version
 * mit den genutzten Min und Max Werten.
 * @param dataArray Ein Array mit Item, dass die Eigenschaften "label", "feature" besitzt
 * @returns {*}
 */
function getNormalizedData(dataArray) {
    return tf.tidy(() => {
        // Durchmischen, trennen des DataSets
        const shuffledArray = getShuffledFeatureAndLabelColumn(dataArray);
        const featureTensor = tf.tensor2d(shuffledArray.feature, [shuffledArray.feature.length, 1]);
        const labelTensor = tf.tensor2d(shuffledArray.label, [shuffledArray.feature.length, 1]);

        // Normalisieren von Features und Labels
        const normalisierungFeatures = normalizeTensor(featureTensor);
        const normalizedFeatures = normalisierungFeatures.normierterTensor;
        const minFeature = normalisierungFeatures.min;
        const maxFeature = normalisierungFeatures.max;

        const normalisierungLabels = normalizeTensor(labelTensor);
        const normalizedLabels = normalisierungLabels.normierterTensor;
        const minLabel = normalisierungLabels.min;
        const maxLabel = normalisierungLabels.max;

        return workDataSet = {
            shuffledArray: shuffledArray,
            featureTensor: featureTensor,
            labelTensor: labelTensor,
            //normalisierungFeatures: normalisierungFeatures,
            normalizedFeatures: normalizedFeatures,
            minFeature: minFeature,
            maxFeature: maxFeature,
            //normalisierungLabels: normalisierungLabels,
            normalizedLabels: normalizedLabels,
            minLabel: minLabel,
            maxLabel: maxLabel,
        };
    });
}

// Helper functions used for sampling

/**
 * Erzeugt eine zufällige, gleichberteilte Nummer im Bereich min bis max
 * @param min Die untere Grenze
 * @param max Die obere Grenze
 * @returns {*} Die Zufallszahl
 */
function getUniformDistributedRandomNumber(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * Fügt dem übergebenen Wert ein Rauschen, basierend auf einen Mittelwert und Varianz hinzu
 * @param label Der zu überlagernde Wert
 * @param mean Das zu verwendende Mittel
 * @param variance Die zu verwendende Varianz
 * @returns {*} Der überlagerte Wert
 */
function addGaussianNoise(label, mean, variance) {
    const labelNoisy = label * Math.sqrt(variance) + mean;
    return labelNoisy;
}

/**
 * Berechnet die Summe des übergebenen, zweidimensionalen Arrays von Zahlen
 * @param dataArray Das Array, über das die Summe berechnet werden soll
 * @returns {number} Die berechnete Summe des Arrays
 */
function calcSum(dataArray) {
    let total = 0;

    for (let i = 0; i < dataArray.length; i++) {
        total += dataArray[i].label;
    }

    return total;
}

/**
 * Berechnet den Mittelwert des übergebenen, zweidimensionalen Arrays von Zahlen
 * @param dataArray Das Array, über das der Mittelwert berechnet werden soll
 * @returns {number} Den berechneten Mittelwert des Arrays
 */
function calcMean(dataArray) {
    let sum = calcSum(dataArray);
    return sum / dataArray.length;
}

/**
 * Überlagert ein übergebenes Array mit einem Rauschen in vorgegebener Varianz
 * @param dataArray Das zu bearbeitende Array
 * @param noise Die anzuwendende Varianz
 * @returns {*} Das mit einem Rauschen überlagerte Array
 */
function getDataArrayWithNoise(dataArray, noise) {
    const mean = calcMean(dataArray);

    for (let item of dataArray) {
        let newLabel = addGaussianNoise(item.label, mean, noise);
        item.label = newLabel;
    }

    return dataArray;
}

/**
 * Erzeugt ein Array mit gleichverteilten, zufälligen Zahlen
 * @param countNumbers Die ANzahl der zu erzeugenden Zahlen
 * @returns {*[]} Ein Array mit Zufallszahlen
 */
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

/**
 * Erzeugt ein Array fortlaufender im angegebenen Bereich und der vorgegebenen Schrittweite
 * @param min Der Startwert
 * @param max Der Endwert
 * @param step Die Schrittweite
 * @returns {*[]} Ein Array mit den Zahlen in aufsteigender Reihenfolge
 */
function getFullDataArray(min, max, step) {
    let dataArray = [];

    for (let i = min; i < max; i = i + step) {
        let feature = i;
        let label = (feature + 0.8) * (feature - 0.2) * (feature - 0.3) * (feature - 0.6);

        dataArray.push({
            feature: feature,
            label: label,
        });
    }

    return dataArray;
}
