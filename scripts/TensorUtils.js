//////////////////////////////////////////////////////////////////////////////////////////
////
//// Beinhaltet Funktionen für den Umgang mit Tensoren und der Normalisierung / Denormalisierung
//// von Daten
////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * Mischt ein Array mit dem Fisher-Yates Algorithmus via TF durch und separiert die Feature und Label
 * Eigenschaft aus dem übergebenen Arrays in zwei separate Arrays
 * @param arr Ein Array mit Elementen (JSON) mit den gesuchten Werten
 * @returns {{features: *[], labels: *[]}} Features und Labels in zwei getrennten Arrays
 */
function getShuffledFeatureAndLabelArray(arr) {
    tf.util.shuffle(arr);

    const featureArray = arr.map(d => d.feature);
    const labelArray = arr.map(d => d.label);

    return {
        featureArray,
        labelArray
    };
}

/**
 * Normalisiert den übergebenen Tensor mit min-max-scaling - inputTensor.sub(min).div(max.sub(min)) - und
 * gibt den normierten Tensor plus das Minimum und Maximum zurück. Durch Übergabe von Min und Max Werten,
 * werden diese genutzt. Sind diese 0, so werden die Werte aus den übergebenen Daten ermittelt.
 * @param inputTensor Der zu normalisierende Tensor
 * @param prevMax Wenn >0, Vorrang vor ermittelten Max
 * @param prevMin Wenn >0, Vorrang vor ermitteten Min
 * @returns {{normalizedTensor: *, min: *, max: *}}
 */
function normalizeTensor(inputTensor, prevMax = 0, prevMin = 0) {
    return tf.tidy(() => {
        const max = prevMax || inputTensor.max();
        const min = prevMin || inputTensor.min();
        const normalizedTensor = inputTensor.sub(min).div(max.sub(min));

        return {
            normalizedTensor,
            max,
            min
        };
    });
}

/**
 * Denormalisiert den übergebenen Tensor mit min-max-scaling - inputTensor.mul(max.sub(min)).add(min) und
 * gibt den denormalisierten Tensor zurück.
 * @param inputTensor Der zu denormalisierende Tensor
 * @param min Der bei der Normalisierung genutzte Minwert
 * @param max Der bei der Normalisierung genutzte Maxwert
 * @returns {*} Ein neuer Tensor mit den ursprünglichen Werten
 */
function denormalizeTensor(inputTensor, min, max) {
    return tf.tidy(() => {
        return inputTensor.mul(max.sub(min)).add(min);
    });
}

/**
 * Erzeugt auf Basis des übergebenen Arrays normalisierte Tensoren für Feature und Label und ermittelt
 * die zugehörigen Min und Max Werte
 * @param dataArray Ein Array mit Item, dass die Eigenschaften "label", "feature" besitzt
 * @returns {*|(function(*, *): *)}
 */
function getNormalizedData(dataArray) {
    return workDataSet = tf.tidy(() => {

        // Durchmischen, trennen des DataSets
        const shuffledArray = getShuffledFeatureAndLabelArray(dataArray);

        // Trennen von feataure und label
        const featureTensor = tf.tensor2d(shuffledArray.featureArray, [shuffledArray.featureArray.length, 1]);
        const labelTensor = tf.tensor2d(shuffledArray.labelArray, [shuffledArray.labelArray.length, 1]);

        // Normalisieren von Features und Labels
        const normalisierungFeatures = normalizeTensor(featureTensor);
        const normalizedFeatures = normalisierungFeatures.normalizedTensor;
        const minFeature = normalisierungFeatures.min;
        const maxFeature = normalisierungFeatures.max;

        const normalisierungLabels = normalizeTensor(labelTensor);
        const normalizedLabels = normalisierungLabels.normalizedTensor;
        const minLabel = normalisierungLabels.min;
        const maxLabel = normalisierungLabels.max;

        // Rückgabe aller normierten Werte
        return {
            shuffledArray,
            normalizedFeatures,
            normalizedLabels,
            minFeature,
            maxFeature,
            minLabel,
            maxLabel,
        };
    });
}