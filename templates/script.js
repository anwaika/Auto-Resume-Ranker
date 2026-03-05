async function rankResumes() {

    const fileInput = document.getElementById("resumeUpload");
    const files = fileInput.files;

    if (files.length === 0) {
        alert("Please upload resumes");
        return;
    }

    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        formData.append("resumes", files[i]);
    }

    const response = await fetch("/rank", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    const resultDiv = document.getElementById("results");
    resultDiv.innerHTML = "";

    data.results.forEach((resume, index) => {
        resultDiv.innerHTML += `<p>${index + 1}. ${resume.name} - Score: ${resume.score}</p>`;
    });
}