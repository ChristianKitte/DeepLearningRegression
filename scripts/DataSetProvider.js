//////////////////////////////////////////////////////////////////////////////////////////
////
//// Hält alle Funktionalität zur Auswahl und Erzeugung eines DataArrays für die Anwendung
//// sowie das gerade aktuelle Array (currentRandomDataArray).
////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * Das aktuell gültige DataSet
 * @type {*[]}
 */
let currentRandomDataArray = [];
/**
 * Das DropDown Element zur Auswahl des DataSets
 * @type {HTMLElement}
 */
let dataArrayDropDow = document.getElementById("data-array");

/**
 * Handelt das Event OnChange des Dropdown für die Auswahl des DataSets
 * @param event
 */
dataArrayDropDow.onchange = function (event) {
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

    let dataArray = getUniformRandomDataArray(countDataArray, -1, 1);
    let dataArrayNoisy = getDataArrayWithNoise(dataArray, 0.3);

    currentRandomDataArray = dataArrayNoisy;

    DrawGraph("dataSetGraphFunction", "Funktionswerte der Rohdaten", dataArrayNoisy);
};