alert("This tool provides preliminary screening only. Not a medical diagnosis.")
const imageInput = document.getElementById('imageUpload');
  const imagePreview = document.getElementById('imagePreview');
  const clearBtn = document.getElementById('clearImageBtn');

  imageInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      imagePreview.src = ev.target.result;
      imagePreview.style.display = 'block';
      clearBtn.style.display = 'block';
    };
    reader.readAsDataURL(file);
  });

  clearBtn.addEventListener('click', () => {
    imageInput.value = '';
    imagePreview.style.display = 'none';
    clearBtn.style.display = 'none';
  });

  document.getElementById('analysisForm').addEventListener('submit', async e => {
  e.preventDefault();

  document.getElementById('loading').style.display = 'flex';

  const formData = new FormData();
  formData.append("image", document.getElementById("imageUpload").files[0]);
  formData.append("age", document.getElementById("ageInput").value);
  formData.append("gender", document.getElementById("genderSelect").value);
  formData.append("location", document.getElementById("locationSelect").value);

  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  });

  const result = await res.json();
  console.log("BACKEND RESPONSE:", result);


  setTimeout(() => {
    document.getElementById('loading').style.display = 'none';
    showResult(result);
  }, 1200);
});


function showResult(res) {
  const list = ["bcc", "mel"];

  let result;

  if (list.includes(res.cancer_type)) {
    result = "You may have Cancer";
  } else {
    result = "You do not have Cancer";
  }

  document.getElementById('resType').textContent = result;
  document.getElementById('resType').style.paddingLeft = "130px";
  document.getElementById('resType').style.color = "#3d3631";
  document.getElementById('resExplanation').textContent = res.explanation;

  const risk = document.getElementById('resRisk');
  risk.textContent = res.risk + " risk";
  risk.className = 'risk-tag risk-' + res.risk;

  document.getElementById('resultModal').style.display = 'flex';
}


  function closeResult() {
    document.getElementById('resultModal').style.display = 'none';
  }