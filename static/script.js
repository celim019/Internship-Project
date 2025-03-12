window.onload = function() {
    let form = document.getElementById("uploadForm");
    let imageInput = document.getElementById("imageInput");
    let preview = document.getElementById("preview");

    if (form) {
        form.onsubmit = function(event) {
            event.preventDefault(); // Stop form submission
            uploadImage();
        };
    }

    // Preview uploaded image
    imageInput.addEventListener("change", function(event) {
        let file = event.target.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    });
};

function uploadImage() {
    let input = document.getElementById("imageInput");
    let file = input.files[0];

    if (!file) {
        alert("⚠️ Please upload an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerText = "❌ Error: " + data.error;
        } else {
            document.getElementById("result").innerText = "✅ Predicted Digit: " + data.prediction;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "⚠️ Error processing image.";
    });
}
