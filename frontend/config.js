const localApi =
  window.location.port === "5173"
    ? "http://127.0.0.1:8000"
    : "https://isl-vision-api.onrender.com";

window.ISL_API_BASE_URL = window.ISL_API_BASE_URL || localApi;