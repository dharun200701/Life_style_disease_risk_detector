document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("predictionForm");
    const button = document.getElementById("predictBtn");

    if (form && button) {

        form.addEventListener("submit", function () {

            button.disabled = true;

            button.innerHTML = `
                <span>
                    <i class="fa-solid fa-spinner fa-spin"></i>
                    Analyzing Health Data...
                </span>
                <i class="fa-solid fa-hourglass-half"></i>
            `;

        });

    }

});