import { useState } from "react";
import {
  Sparkles,
  Upload,
  Image as ImageIcon,
  Brain,
  Cpu,
  Zap,
  X,
  Copy,
  Check,
} from "lucide-react";

import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleImage = (file) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please select an image file.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setCaption("");
    setCopied(false);
  };

  const handleFileChange = (event) => {
    handleImage(event.target.files[0]);
  };

  const removeImage = () => {
    setImage(null);
    setPreview(null);
    setCaption("");
    setCopied(false);
  };

  const generateCaption = async () => {
    if (!image) return;

    setLoading(true);
    setCaption("");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/caption`, {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Failed to generate caption");
      }

      const data = await response.json();

      setCaption(data.caption);
    } catch (error) {
      console.error(error);
      setCaption(
        "Unable to generate caption. Make sure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const copyCaption = async () => {
    if (!caption) return;

    await navigator.clipboard.writeText(caption);

    setCopied(true);

    setTimeout(() => {
      setCopied(false);
    }, 1500);
  };

  return (
    <div className="app">

      {/* Background decorations */}
      <div className="glow glow-one"></div>
      <div className="glow glow-two"></div>

      {/* Navbar */}
      <nav className="navbar">

        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={20} />
          </div>

          <span>VisionCaption AI</span>
        </div>

        <div className="nav-badge">
          <span className="status-dot"></span>
          CNN + LSTM
        </div>

      </nav>


      {/* Hero */}
      <main>

        <section className="hero">

          <div className="hero-badge">
            <Sparkles size={16} />
            AI IMAGE UNDERSTANDING
          </div>

          <h1>
            Turn images into
            <span> intelligent captions.</span>
          </h1>

          <p>
            Upload an image and let our CNN + LSTM architecture
            understand the scene and generate a natural-language
            description.
          </p>

        </section>


        {/* Main Card */}
        <section className="workspace">

          {/* Upload */}
          <div className="panel">

            <div className="panel-header">

              <div>
                <h2>Upload Image</h2>
                <p>Choose an image to analyze</p>
              </div>

              <ImageIcon size={22} />

            </div>


            {!preview ? (

              <label className="upload-area">

                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  hidden
                />

                <div className="upload-icon">
                  <Upload size={28} />
                </div>

                <h3>Drop your image here</h3>

                <p>
                  or click to browse from your computer
                </p>

                <span>
                  JPG, JPEG, PNG • Max 10MB
                </span>

              </label>

            ) : (

              <div className="preview-container">

                <img
                  src={preview}
                  alt="Preview"
                  className="preview-image"
                />

                <button
                  className="remove-button"
                  onClick={removeImage}
                >
                  <X size={18} />
                </button>

              </div>

            )}


            {image && (

              <button
                className="generate-button"
                onClick={generateCaption}
                disabled={loading}
              >

                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing image...
                  </>
                ) : (
                  <>
                    <Sparkles size={19} />
                    Generate Caption
                  </>
                )}

              </button>

            )}

          </div>


          {/* Result */}
          <div className="panel result-panel">

            <div className="panel-header">

              <div>
                <h2>AI Caption</h2>
                <p>Generated by your trained model</p>
              </div>

              <Brain size={22} />

            </div>


            <div className="result-area">

              {!caption && !loading && (

                <div className="empty-result">

                  <div className="result-icon">
                    <Brain size={30} />
                  </div>

                  <h3>Your caption will appear here</h3>

                  <p>
                    Upload an image and click
                    <strong> Generate Caption</strong>.
                  </p>

                </div>

              )}


              {loading && (

                <div className="loading-state">

                  <div className="loading-orbit">
                    <Brain size={32} />
                  </div>

                  <h3>Understanding your image...</h3>

                  <p>
                    ResNet50 is extracting visual features
                    and the LSTM is generating your caption.
                  </p>

                </div>

              )}


              {caption && !loading && (

                <div className="caption-result">

                  <div className="quote-mark">“</div>

                  <p className="caption-text">
                    {caption}
                  </p>

                  <div className="result-actions">

                    <button
                      onClick={copyCaption}
                      className="copy-button"
                    >
                      {copied ? (
                        <>
                          <Check size={16} />
                          Copied
                        </>
                      ) : (
                        <>
                          <Copy size={16} />
                          Copy Caption
                        </>
                      )}
                    </button>

                  </div>

                </div>

              )}

            </div>

          </div>

        </section>


        {/* Architecture */}
        <section className="architecture">

          <div className="section-title">

            <span>HOW IT WORKS</span>

            <h2>
              From pixels to language
            </h2>

          </div>


          <div className="architecture-grid">

            <div className="architecture-card">

              <div className="architecture-icon">
                <Upload size={22} />
              </div>

              <div>
                <span>01</span>
                <h3>Image Input</h3>
                <p>
                  Your image is resized and normalized
                  before entering the neural network.
                </p>
              </div>

            </div>


            <div className="architecture-card">

              <div className="architecture-icon">
                <Cpu size={22} />
              </div>

              <div>
                <span>02</span>
                <h3>CNN Encoder</h3>
                <p>
                  ResNet50 extracts 2048-dimensional
                  visual features from the image.
                </p>
              </div>

            </div>


            <div className="architecture-card">

              <div className="architecture-icon">
                <Brain size={22} />
              </div>

              <div>
                <span>03</span>
                <h3>LSTM Decoder</h3>
                <p>
                  The LSTM converts visual features
                  into a sequence of meaningful words.
                </p>
              </div>

            </div>


            <div className="architecture-card">

              <div className="architecture-icon">
                <Zap size={22} />
              </div>

              <div>
                <span>04</span>
                <h3>Caption</h3>
                <p>
                  The generated tokens are converted
                  into a natural-language caption.
                </p>
              </div>

            </div>

          </div>

        </section>

      </main>


      <footer>
        <span>VisionCaption AI</span>
        <span>Built with React • FastAPI • PyTorch</span>
      </footer>

    </div>
  );
}

export default App;