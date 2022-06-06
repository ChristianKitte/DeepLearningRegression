/**
 * Hält Beschreibungen des aktuellen neuronalen Netzes. Diese können zur information oder
 * der Speicherung des Aufbaus verwendet werden.
 */
class ModelDescription {
    /**
     * @param inputUnits Die Anzahl der Eingaben und Ausgaben (wenn Flatten = False ist)
     * @param hiddenLayers Die Aazuahl der versteckten Schichten
     * @param hiddenUnits Die Anzahl der Perzeptoren je versteckter Schicht
     * @param useBias Die Angabe, ob Bias genutzt werden soll
     * @param activation Die zu verwendende Aktivierungsfunktion : (elu, hardSigmoid, linear, relu, relu6, selu,
     * sigmoid, softmax, softplus, softsign, tanh, swish, mish)
     * @param optimizerInstance Eine Instanz des zu verwendenden Optimierer an
     * @param lossFunction Die zu verwendende Verlustfunktion
     * @param metricsArray Ein Array der zu erzeugenden Metriken
     */
    constructor(inputUnits,
                hiddenLayers,
                hiddenUnits,
                useBias,
                activation,
                optimizerInstance,
                lossFunction,
                metricsArray) {

        this.inputUnits = inputUnits;
        this.hiddenLayers = hiddenLayers;
        this.hiddenUnits = hiddenUnits;
        this.useBias = useBias;
        this.activation = activation;
        this.optimizerInstance = optimizerInstance;
        this.lostFunktion = lossFunction;
        this.metricsArray = metricsArray;
    }

    /**
     * Liefert den JSON des aktuell definierten neuronalen Netzes zurück.
     * @returns {{
     * hiddenUnits: *,
     * lostFunktion: (getJSON.lostFunktion|*),
     * inputUnits: (number|*), useBias,
     * optimizerInstance: (getJSON.optimizerInstance|*),
     * hiddenLayers: (number|*),
     * activation,
     * metricsArray: (getJSON.metricsArray|*)}}
     */
    function

    toJSON() {
        return {
            inputUnits: this.inputUnits,
            hiddenLayers: this.hiddenLayers,
            hiddenUnits: this.hiddenUnits,
            useBias: this.useBias,
            activation: this.activation,
            optimizerInstance: this.optimizerInstance,
            lostFunktion: this.lostFunktion,
            metricsArray: this.metricsArray,
        }
    }
}