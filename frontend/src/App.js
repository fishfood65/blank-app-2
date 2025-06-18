import React, { useState } from "react";
import axios from "axios";

function App() {
  const [inputs, setInputs] = useState({
    city: "",
    zip_code: "",
    internet_provider: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setInputs((prev) => ({ ...prev, [name]: value }));
  };

  const handleGenerate = async () => {
    try {
      const response = await axios.post(
        "/generate-runbook/",
        {
          section: "utilities",
          inputs,
        },
        { responseType: "blob" }
      );

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "utilities_runbook.docx");
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      alert("⚠️ Error generating runbook");
      console.error(err);
    }
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "600px" }}>
      <h2>🔌 Utilities Setup</h2>

      <input
        name="city"
        placeholder="City"
        value={inputs.city}
        onChange={handleChange}
        required
      /><br />

      <input
        name="zip_code"
        placeholder="ZIP Code"
        value={inputs.zip_code}
        onChange={handleChange}
        required
      /><br />

      <input
        name="internet_provider"
        placeholder="Internet Provider (optional)"
        value={inputs.internet_provider}
        onChange={handleChange}
      /><br />

      <button style={{ marginTop: "1rem" }} onClick={handleGenerate}>
        ⚙️ Generate Utilities Runbook
      </button>
    </div>
  );
}

export default App;

