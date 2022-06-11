//////////////////////////////////////////////////////////////////////////////////////////
////
//// Hält alle Anwendungseinstellungen und -daten. Handelt alle Ereignisse, die Daten
//// der Anwendung abändern ohne Button Click
////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * Das aktuelle neuronale Netz
 */
let currentNeuralNet;
/**
 * Die aktuellen zum training verwendeten, normierten Daten
 */
let currentTrainData;

// Ausgabecontainer

/**
 * Der Ausgabecontainer für die Dokumentation des Modells
 * @type {string}
 */
const DOKU_MODEL = "dokuNetz"
/**
 * Der Ausgabecontainer für die Dokumentation des Trainings
 * @type {string}
 */
const DOKU_TRAIN = "dokuTrain"
/**
 * Der Ausgabecontainer für die Dokumentation des Tests
 * @type {string}
 */
const DOKU_TEST = "dokuTest"

// Definition der Beispielnetze

/**
 * Definition eines überangepasten Modells
 * @type {{Model: {hiddenUnits: number, lostFunktion: string, inputUnits: number, useBias: boolean, Flatten: boolean, optimizerInstance: string, hiddenLayers: number, activation: string, metricsArray: string}, training: {batchSize: number, dataset: number, epochs: number}}}
 */
const OVER_FITTED = {
    Model: {
        units: 48,
        layers: 13,
        activation: 2,
        optimizer: 4,
        lost: 1
    },
    training: {
        dataset: 5,
        batchSize: 48,
        epochs: 300
    }
};

/**
 * Definition eines unterangepasten Modells
 * * @type {{Model: {hiddenUnits: number, lostFunktion: string, inputUnits: number, useBias: boolean, Flatten: boolean, optimizerInstance: string, hiddenLayers: number, activation: string, metricsArray: string}, training: {batchSize: number, dataset: number, epochs: number}}}
 *  */
const UNDER_FITTED = {
    Model: {
        units: 4,
        layers: 2,
        activation: 2,
        optimizer: 4,
        lost: 1
    },
    training: {
        dataset: 5,
        batchSize: 25,
        epochs: 100
    }
};

/**
 * Definition eines gut angepassten Modells
 * * @type {{Model: {hiddenUnits: number, lostFunktion: string, inputUnits: number, useBias: boolean, Flatten: boolean, optimizerInstance: string, hiddenLayers: number, activation: string, metricsArray: string}, training: {batchSize: number, dataset: number, epochs: number}}}
 *  */
const BEST_FITTED = {
    Model: {
        units: 32,
        layers: 13,
        activation: 2,
        optimizer: 4,
        lost: 1
    },
    training: {
        dataset: 5,
        batchSize: 32,
        epochs: 300
    }
};

// Defnition der Variablen

/**
 * Die Anzahl der zu verwendenden Hidden Layer
 * @type {number}
 */
let countHiddenLayer = 1;

/**
 * Die Anzahl der zu verwendenden neuronen je Layer
 * @type {number}
 */
let countUnits = 1;

/**
 * Welche Aktivierungsfunktion soll verwendet werden?
 * 1: sigmoid, 2:relu, 3:tanh
 * @type {number}
 */
let activationSelection = 0;

/**
 * Welche Verlustfunktion soll verwendet werden
 * 1: mse, 2:absolut
 * @type {number}
 */
let lossSelection = 0;

/**
 * Welcher Optimierer soll verwendet werden
 * 1: sgd, 2: momentum, 3: adam
 * @type {number}
 */
let optimizerSelection = 0;

/**
 * Gibt die Größe eines Batches beim Training an
 * @type {number}
 */
let countBatch = 1;

/**
 * Gibt die Anzahl der Epochen für das Training an
 * @type {number}
 */
let countEpoch = 1;

// Auslesen der Slider

/**
 * OnInput Event Handler für countHiddenLayer
 */
d3.select('#count-layer')
    .on("input", function () {
        countHiddenLayer = +this.value;
        let txt = "Aktueller Wert: " + countHiddenLayer.toString() + " Layer";
        d3.select('#count-layer-string').text(txt);
    });

/**
 * OnInput Event Handler für countUnits
 */
d3.select('#count-neuron')
    .on("input", function () {
        countUnits = parseInt(+this.value);
        let txt = "Aktueller Wert: " + countUnits.toString() + " Units";
        d3.select('#count-neuron-string').text(txt);
    });

/**
 * OnInput Event Handler für countBatchSize
 */
d3.select('#count-batch')
    .on("input", function () {
        countBatch = parseInt(+this.value);
        let txt = "Aktueller Wert: " + countBatch.toString();
        d3.select('#count-batch-string').text(txt);
    });

/**
 * OnInput Event Handler für countEpoch
 */
d3.select('#count-epoch')
    .on("input", function () {
        countEpoch = parseInt(+this.value);
        let txt = "Aktueller Wert: " + countEpoch.toString();
        d3.select('#count-epoch-string').text(txt);
    });

// Auslesen der DropDowns

/**
 * OnChange Event Handler für Aktivierungstyp
 */
d3.select('#activation-type')
    .on("change", function () {
        activationSelection = +this.value;
    })

/**
 * OnChange Event Handler für Verlustfunktion
 */
d3.select('#loss-type')
    .on("change", function () {
        lossSelection = +this.value;
    })

/**
 * OnChange Event Handler für den Optimierer
 */
d3.select('#optimizer-type')
    .on("change", function () {
        optimizerSelection = +this.value;
    })

// Methoden

/**
 * Gibt den zu activationSelection passenden String zurück. Wurde keine auswahl getroffen, so
 * wird sigmoid verwendet
 * 1: sigmoid, 2:relu, 3:tanh, 4:softmax, 5: linear
 */
function getActivationSelectionString() {
    if (activationSelection === 1) {
        return 'sigmoid';
    } else if (activationSelection === 2) {
        return 'relu';
    } else if (activationSelection === 3) {
        return 'tanh';
    } else if (activationSelection === 4) {
        return 'softmax';
    } else if (activationSelection === 5) {
        return 'linear';
    } else {
        return 'sigmoid';
    }
}

/**
 * Gibt den zu activationSelection passenden String zurück. Wurde keine Auswahl getroffen,
 * so wird mse verwendet
 * 1: mse, 2:absolut
 */
function getLossSelectionString() {
    if (lossSelection === 1) {
        return tf.losses.meanSquaredError;
    } else if (lossSelection === 2) {
        return tf.losses.meanAbsoluteError;
    } else {
        return tf.losses.meanSquaredError;
    }
}

/**
 * Gibt die anzuwendende Metric auf Basis der ausgewählten activationSelection zurück. Wurde keine Auswahl getroffen,
 * so wird mse verwendet
 * 1: mse, 2:absolut
 */
function getMetricsByLossSelection() {
    let retVal;
    if (lossSelection === 1) {
        retVal = tf.metrics.meanSquaredError;
    } else {
        retVal = tf.metrics.meanSquaredError;
    }

    return [retVal];
}

/**
 * Gibt den zu optimizerSelection passenden String zurück. Wurde keine Auswahl getroffen,
 * so wird adam verwendet
 * 1: sgd, 2: momentum, 3: adam, 4: adamax, 5:rmsprop
 * https://peltarion.com/knowledge-center/documentation/modeling-view/run-a-model/optimizers/adamax
 * https://www.geeksforgeeks.org/tensorflow-js-tf-train-momentum-function/
 */
function getOptimizerInstance() {
    if (optimizerSelection === 1) {
        return tf.train.sgd(0.01);
    } else if (optimizerSelection === 2) {
        const learningRate = 0.01;
        const momentum = 0.1;
        const useNestrov = true;

        return tf.train, momentum(learningRate, momentum, useNestrov)
    } else if (optimizerSelection === 2) {
        return tf.train.rmsprop(0.01);
    } else if (optimizerSelection === 3) {
        return tf.train.adam();
    } else if (optimizerSelection === 4) {
        return tf.train.adamax(0.01);
    } else if (optimizerSelection === 5) {
        return tf.train.rmsprop(0.001);
    }
}

/**
 * Setzt die initialen Werte der Oberfläche basierend auf den aktuellen Werten nach Abschluss des Ladevorgangs
 */
function setInitialValue() {
    // Auslesen der Slider
    countHiddenLayer = parseInt(document.getElementById('count-layer').getAttribute('value'));
    let txt1 = "Aktueller Wert: " + countHiddenLayer.toString() + " Layer";
    d3.select('#count-layer-string').text(txt1);

    countUnits = parseInt(document.getElementById('count-neuron').getAttribute('value'));
    let txt2 = "Aktueller Wert: " + countUnits.toString() + " Units";
    d3.select('#count-neuron-string').text(txt2);

    countBatch = parseInt(document.getElementById('count-batch').getAttribute('value'));
    let txt4 = "Aktueller Wert: " + countBatch.toString();
    d3.select('#count-batch-string').text(txt4);

    countEpoch = parseInt(document.getElementById('count-epoch').getAttribute('value'));
    let txt5 = "Aktueller Wert: " + countEpoch.toString();
    d3.select('#count-epoch-string').text(txt5);

    // Auslesen der DropDowns
    activationSelection = document.getElementById("activation-type").selectedIndex;
    lossSelection = document.getElementById("loss-type").selectedIndex;
    optimizerSelection = document.getElementById("optimizer-type").selectedIndex;
}
