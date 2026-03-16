const API_URL = "http://localhost:8000/summarize";
const JOBS_URL = "http://localhost:8000/jobs";
const TRAIN_URL = "http://localhost:8000/jobs/train";
const EVAL_URL = "http://localhost:8000/jobs/evaluate";

const inputEl = document.getElementById("input-text");
const maxLengthEl = document.getElementById("max-length");
const temperatureEl = document.getElementById("temperature");
const summarizeBtn = document.getElementById("summarize-btn");
const summaryOutput = document.getElementById("summary-output");
const statusChip = document.getElementById("status-chip");
const tokensUsed = document.getElementById("tokens-used");
const latency = document.getElementById("latency");
const device = document.getElementById("device");
const trainBtn = document.getElementById("train-btn");
const evalBtn = document.getElementById("eval-btn");
const trainStatus = document.getElementById("train-status");
const evalStatus = document.getElementById("eval-status");
const trainMessage = document.getElementById("train-message");
const evalMessage = document.getElementById("eval-message");
const trainLogs = document.getElementById("train-logs");
const evalLogs = document.getElementById("eval-logs");

const setStatus = (label, busy = false) => {
    statusChip.textContent = label;
    statusChip.classList.toggle("busy", busy);
};

const summarize = async () => {
    const text = inputEl.value.trim();
    if (!text) {
        summaryOutput.textContent = "Please provide medical text to summarize.";
        return;
    }

    setStatus("Generating...", true);
    summaryOutput.textContent = "Synthesizing summary with entity focus...";
    summarizeBtn.disabled = true;

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                max_length: Number(maxLengthEl.value),
                temperature: Number(temperatureEl.value),
            }),
        });

        if (!response.ok) {
            throw new Error("API request failed");
        }

        const data = await response.json();
        summaryOutput.textContent = data.summary;
        tokensUsed.textContent = data.tokens_used;
        latency.textContent = `${data.generation_time_s.toFixed(2)}s`;
        device.textContent = data.device;
        setStatus("Ready", false);
    } catch (error) {
        console.error(error);
        summaryOutput.textContent = "Unable to generate summary. Ensure the backend is running.";
        setStatus("Error", false);
    } finally {
        summarizeBtn.disabled = false;
    }
};

summarizeBtn.addEventListener("click", summarize);

// Allow Cmd/Ctrl+Enter to trigger summarization
inputEl.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        summarize();
    }
});

const updateJobCard = (elements, job) => {
    const { status, message, logs_tail: logs } = job;
    elements.status.textContent = status ?? "unknown";
    elements.status.classList.toggle("busy", status === "running");
    elements.button.disabled = status === "running";
    elements.message.textContent = message ?? "";
    elements.logs.textContent = logs ?? "No logs yet.";
};

const refreshJobs = async () => {
    try {
        const response = await fetch(JOBS_URL);
        if (!response.ok) throw new Error("Failed to fetch job status");
        const data = await response.json();
        if (data.train) {
            updateJobCard(
                {
                    status: trainStatus,
                    button: trainBtn,
                    message: trainMessage,
                    logs: trainLogs,
                },
                data.train
            );
        }
        if (data.evaluate) {
            updateJobCard(
                {
                    status: evalStatus,
                    button: evalBtn,
                    message: evalMessage,
                    logs: evalLogs,
                },
                data.evaluate
            );
        }
    } catch (error) {
        console.error(error);
    }
};

const triggerJob = async (endpoint, button) => {
    button.disabled = true;
    try {
        const response = await fetch(endpoint, {
            method: "POST",
        });
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Job trigger failed");
        }
    } catch (error) {
        alert(error.message);
    } finally {
        setTimeout(() => {
            button.disabled = false;
            refreshJobs();
        }, 500);
    }
};

trainBtn.addEventListener("click", () => triggerJob(TRAIN_URL, trainBtn));
evalBtn.addEventListener("click", () => triggerJob(EVAL_URL, evalBtn));

refreshJobs();
setInterval(refreshJobs, 5000);

// Three.js neon scene
const canvas = document.getElementById("bg-canvas");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);
camera.position.z = 25;

const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
});
renderer.setSize(window.innerWidth, window.innerHeight);

const addHelix = () => {
    const group = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({
        color: 0x00ff99,
        emissive: 0x007744,
        metalness: 0.8,
        roughness: 0.2,
    });

    const geometry = new THREE.TorusKnotGeometry(3.5, 1.1, 200, 32);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 0, 0);
    group.add(mesh);

    const ringGeometry = new THREE.TorusGeometry(6, 0.08, 16, 100);
    const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ff99,
        transparent: true,
        opacity: 0.35,
    });

    for (let i = 0; i < 3; i += 1) {
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.rotation.x = Math.PI / (i + 2);
        ring.rotation.y = (i * Math.PI) / 3;
        group.add(ring);
    }

    scene.add(group);
    return group;
};

const neonHelix = addHelix();

const light = new THREE.PointLight(0x00ff99, 2, 100);
light.position.set(10, 10, 10);
scene.add(light);

const ambient = new THREE.AmbientLight(0x003322, 0.8);
scene.add(ambient);

const animate = () => {
    requestAnimationFrame(animate);
    neonHelix.rotation.x += 0.002;
    neonHelix.rotation.y += 0.003;
    renderer.render(scene, camera);
};
animate();

window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

