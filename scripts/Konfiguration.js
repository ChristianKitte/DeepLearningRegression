//const LOCAL_HOST = "https://deep-learning.ckitte.de/ea3/";
const LOCAL_HOST = "./";
const LOCAL_DATA = LOCAL_HOST + "data/";
const PREDEFINED_DATASET_INDEX = "4";

// Halten der aktuell gültigen Netze

/**
 * Das aktuelle neuronale Netz
 */
let currentNeuralNet;
let currentNeuralNetDescriptor;

// Definition der Beispielnetze

/**
 * Definition eines überangepasten Modells
 * @type {{Model: {hiddenUnits: number, lostFunktion: string, inputUnits: number, useBias: boolean, Flatten: boolean, optimizerInstance: string, hiddenLayers: number, activation: string, metricsArray: string}, training: {batchSize: number, dataset: number, epochs: number}}}
 */
const OVER_FITTED = {
    Model: {
        inputUnits: 0,
        hiddenLayers: 0,
        hiddenUnits: 0,
        useBias: true,
        activation: '',
        Flatten: false,
        optimizerInstance: '',
        lostFunktion: '',
        metricsArray: ''
    },
    training: {
        dataset: 0,
        batchSize: 0,
        epochs: 0
    }
};

/**
 * Definition eines unterangepasten Modells
 * * @type {{Model: {hiddenUnits: number, lostFunktion: string, inputUnits: number, useBias: boolean, Flatten: boolean, optimizerInstance: string, hiddenLayers: number, activation: string, metricsArray: string}, training: {batchSize: number, dataset: number, epochs: number}}}
 *  */
const UNDER_FITTED = {
    Model: {
        inputUnits: 0,
        hiddenLayers: 0,
        hiddenUnits: 0,
        useBias: true,
        activation: '',
        Flatten: false,
        optimizerInstance: '',
        lostFunktion: '',
        metricsArray: ''
    },
    training: {
        training: {
            dataset: 0,
            batchSize: 0,
            epochs: 0
        }
    }
};

/**
 * Definition eines gut angepassten Modells
 * * @type {{Model: {hiddenUnits: number, lostFunktion: string, inputUnits: number, useBias: boolean, Flatten: boolean, optimizerInstance: string, hiddenLayers: number, activation: string, metricsArray: string}, training: {batchSize: number, dataset: number, epochs: number}}}
 *  */
const BEST_FITTED = {
    Model: {
        inputUnits: 0,
        hiddenLayers: 0,
        hiddenUnits: 0,
        useBias: true,
        activation: '',
        Flatten: false,
        optimizerInstance: '',
        lostFunktion: '',
        metricsArray: ''
    },
    training: {
        training: {
            dataset: 0,
            batchSize: 0,
            epochs: 0
        }
    }
};

// Defnition der VAriablen

/**
 * Die aktuelle Auswahl eines vordefinierten Netzes
 * @type {number}
 */
let predefinedNet = 2;
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
 * Die zu verwendende Lernrate beim Training
 * @type {number}
 */
let countLearningRate = 1;
/**
 * Gibt an, ob und wie stark das überlagerte Rauschen beim Training sein soll
 * @type {number}
 */
let useNoise = 1;

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

// Auslesen der Optiongruppen

/**
 * Wird aufgerufen, sobald sich die Auswahl eines vordefinierten Netzes ändert. Der
 * Aufruf erfolgt direkt aus dem HTML mit Übergabe einer Nummer.
 * (0 underfit, 1 overfit, 2 best fit)
 * @param value Die getroffene Auswahl
 */
function onPredefinedChanged(value) {
    predefinedNet = value;

    if (value === 0) {
        createUnderfittedFfnn();
    } else if (value === 1) {
        createOverfittedFfnn();
    } else if (value === 2) {
        createBestfittedFfnn();
    }
}

/**
 * Wird aufgerufen, sobald sich die Auswahl eines Rauschen für das Training ändert. Der Aufruf erfolgt
 * direkt aus dem HTML mit Übergabe einer Nummer.
 * (0 kein Rauschen, 1 noise mit 0,1, 2 noise mit 0,2)
 * @param value Die getroffene Auswahl
 */
function onNoiseChanged(value) {
    useNoise = value;
}

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
 * OnInput Event Handler für countLearningRate
 */
d3.select('#count-learningrate')
    .on("input", function () {
        countLearningRate = parseInt(+this.value);
        let txt = "Aktueller Wert: " + countLearningRate.toString();
        d3.select('#count-learningrate-string').text(txt);
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
 * Setzt die initialen Werte nach Abschluss des Ladevorgangs
 */
function setInitialValue() {
    // Auslesen der Optiongruppen
    predefinedNet = getRadioOption("option-predefined");
    useNoise = getRadioOption("option-noise");

    // Auslesen der Slider
    countHiddenLayer = parseInt(document.getElementById('count-layer').getAttribute('value'));
    let txt1 = "Aktueller Wert: " + countHiddenLayer.toString() + " Layer";
    d3.select('#count-layer-string').text(txt1);

    countUnits = parseInt(document.getElementById('count-neuron').getAttribute('value'));
    let txt2 = "Aktueller Wert: " + countUnits.toString() + " Units";
    d3.select('#count-neuron-string').text(txt2);

    countLearningRate = parseInt(document.getElementById('count-learningrate').getAttribute('value'));
    let txt3 = "Aktueller Wert: " + countLearningRate.toString();
    d3.select('#count-learningrate-string').text(txt3);

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

/**
 * Gibt den zu activationSelection passenden String zurück. Wurde keine auswahl getroffen, so
 * wird sigmoid verwendet
 * 1: sigmoid, 2:relu, 3:tanh
 */
function getActivationSelectionString() {
    if (activationSelection === 1) {
        return 'sigmoid';
    } else if (activationSelection === 2) {
        return 'relu';
    } else if (activationSelection === 3) {
        return 'tanh';
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
    } else if (lossSelection === 2) {
        retVal = tf.metrics.meanAbsoluteError;
    } else {
        retVal = tf.metrics.meanSquaredError;
    }

    return [retVal];
}

/**
 * Gibt den zu optimizerSelection passenden String zurück. Wurde keine Auswahl getroffen,
 * so wird adam verwendet
 * 1: sgd, 2: momentum, 3: adam
 */
function getOptimizerInstance() {
    if (optimizerSelection === 1) {
        return tf.train.sgd();
    } else if (optimizerSelection === 2) {
        //return tf.optimizer.momentum(1,1,true);
    } else {
        return tf.train.adam();
    }
}
