/**
 * Das DropDown Element zur Auswahl des DataSets
 * @type {HTMLElement}
 */
let dataSetDropDow = document.getElementById("dataset");

/**
 * Handler für das OnChange Event des Dropdown zur Anzeige der DataSets. Die Funktion wählt je nach Auswahl eines
 * der geladenen DataSets aus und füllt sechs Arrays zu Anzeige.
 *     featureArray = [];       ==> Features
 *     labelArray = [];         ==> Labels
 *     noiseArray01 = [];       ==> Noise mit Standardabweichung von 0,1
 *     noiseArray03 = [];       ==> Noise mit Standardabweichung von 0,3
 *     labelArray01 = [];       ==> Label plus Noise mit Standardabweichung von 0,1
 *     labelArray03 = [];       ==> Label plus Noise mit Standardabweichung von 0,3
 * @param event Das Event
 */
dataSetDropDow.onchange = function (event) {
    featureArray = [];
    labelArray = [];
    noiseArray01 = [];
    noiseArray03 = [];
    labelArray01 = [];
    labelArray03 = [];

    if (this.value === "1") { // 5
        DATA_SET_05.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });
    } else if (this.value === "2") { // 10
        DATA_SET_10.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });
    } else if (this.value === "3") { // 20
        DATA_SET_20.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });
    } else if (this.value === "4") { // 50
        DATA_SET_50.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });
    } else if (this.value === "5") { // 100
        DATA_SET_100.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });
    }

    DrawGraph("dataSetGraphFunction", "Funktionswerte der Rohdaten", featureArray, labelArray);
    DrawGraph("dataSetGraphNoise01", "Rauschen mit Standardabweichung 0,1", featureArray, noiseArray01);
    DrawGraph("dataSetGraphNoise03", "Rauschen mit Standardabweichung 0,3", featureArray, noiseArray03);
    DrawGraph("dataSetGraphFunctionNoise01", "Überlagerte Funktionswerte (0,1)", featureArray, labelArray01);
    DrawGraph("dataSetGraphFunctionNoise03", "Überlagerte Funktionswerte (0,3)", featureArray, labelArray03);
};

