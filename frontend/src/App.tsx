import { useState } from "react";
import "./App.css";

type PredictionResponse = {
  predicted_score?: number;
  model?: ModelKey;
};

type ModelKey = "gradient_boosting" | "random_forest" | "linear_regression";

type ModelOption = {
  value: ModelKey;
  label: string;
  description: string;
  mae: string;
  r2: string;
};

const modelOptions: ModelOption[] = [
  {
    value: "gradient_boosting",
    label: "Gradient Boosting",
    description: "Balanced ensemble",
    mae: "0.57",
    r2: "0.810",
  },
  {
    value: "random_forest",
    label: "Random Forest",
    description: "Flexible ensemble",
    mae: "1.05",
    r2: "0.705",
  },
  {
    value: "linear_regression",
    label: "Linear Regression",
    description: "Fast and transparent",
    mae: "0.41",
    r2: "0.826",
  },
];

function App() {
  const [predictedScore, setPredictedScore] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelKey>("gradient_boosting");

  const [hoursStudied, setHoursStudied] = useState(20);
  const [attendance, setAttendance] = useState(80);
  const [previousScore, setPreviousScore] = useState(70);
  const [sleepHours, setSleepHours] = useState(7);
  const [tutoringSessions, setTutoringSessions] = useState(2);
  const [physicalActivity, setPhysicalActivity] = useState(5);

  const [parentalInvolvement, setParentalInvolvement] = useState("Medium");
  const [accessToResources, setAccessToResources] = useState("Medium");
  const [extracurricular, setExtracurricular] = useState("Yes");
  const [motivation, setMotivation] = useState("Medium");
  const [internetAccess, setInternetAccess] = useState("Yes");
  const [familyIncome, setFamilyIncome] = useState("Medium");
  const [teacherQuality, setTeacherQuality] = useState("Medium");
  const [learningDisabilities, setLearningDisabilities] = useState("No");
  const [parentalEducation, setParentalEducation] = useState("College");
  const [distanceFromHome, setDistanceFromHome] = useState("Moderate");
  const [schoolType, setSchoolType] = useState("Public");
  const [peerInfluence, setPeerInfluence] = useState("Neutral");
  const [gender, setGender] = useState("Male");

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const payload = {
        model: selectedModel,
        Hours_Studied: hoursStudied,
        Attendance: attendance,
        Parental_Involvement: parentalInvolvement,
        Access_to_Resources: accessToResources,
        Extracurricular_Activities: extracurricular,
        Sleep_Hours: sleepHours,
        Previous_Scores: previousScore,
        Motivation_Level: motivation,
        Internet_Access: internetAccess,
        Tutoring_Sessions: tutoringSessions,
        Family_Income: familyIncome,
        Teacher_Quality: teacherQuality,
        Physical_Activity: physicalActivity,
        Learning_Disabilities: learningDisabilities,
        Parental_Education_Level: parentalEducation,
        Distance_from_Home: distanceFromHome,
        School_Type: schoolType,
        Peer_Influence: peerInfluence,
        Gender: gender,
      };

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed with ${response.status}`);
      }

      const result = (await response.json()) as PredictionResponse;

      if (typeof result.predicted_score !== "number") {
        throw new Error("The API returned an invalid prediction");
      }

      setPredictedScore(result.predicted_score);
    } catch (requestError) {
      console.error(requestError);
      setError("Prediction unavailable. Make sure the API is running and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setPredictedScore(null);
    setError(null);
    setSelectedModel("gradient_boosting");
    setHoursStudied(20);
    setAttendance(80);
    setPreviousScore(70);
    setSleepHours(7);
    setTutoringSessions(2);
    setPhysicalActivity(5);
    setParentalInvolvement("Medium");
    setAccessToResources("Medium");
    setExtracurricular("Yes");
    setMotivation("Medium");
    setInternetAccess("Yes");
    setFamilyIncome("Medium");
    setTeacherQuality("Medium");
    setLearningDisabilities("No");
    setParentalEducation("College");
    setDistanceFromHome("Moderate");
    setSchoolType("Public");
    setPeerInfluence("Neutral");
    setGender("Male");
  };

  const scorePercent = Math.min(Math.max(predictedScore ?? 0, 0), 100);
  const scoreTone =
    predictedScore === null
      ? ""
      : predictedScore >= 80
        ? "strong"
        : predictedScore >= 60
          ? "steady"
          : "support";
  const scoreMessage =
    predictedScore === null
      ? "Run a prediction to see the model's estimate."
      : predictedScore >= 80
        ? "A strong academic outlook based on this profile."
        : predictedScore >= 60
          ? "A promising baseline with room to grow."
          : "This profile may benefit from extra support.";
  const selectedModelInfo =
    modelOptions.find((option) => option.value === selectedModel) ?? modelOptions[0];

  const profileHighlights = [
    { label: "Hours studied", value: `${hoursStudied}h` },
    { label: "Attendance", value: `${attendance}%` },
    { label: "Previous score", value: `${previousScore}` },
  ];

  return (
    <div className="app-shell">
      <header className="navbar">
        <a className="brand" href="#top" aria-label="EduPredict home">
          <span className="brand-mark">EP</span>
          <span className="brand-name">EduPredict</span>
        </a>

        <div className="nav-actions">
          <span className="status-pill">
            <span className="status-dot" />
            Model online
          </span>
          <a
            href="https://github.com/"
            target="_blank"
            rel="noreferrer"
            className="github-link"
          >
            <span>View on GitHub</span>
            <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <main id="top" className="container">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">
              <span className="eyebrow-dot" /> Student performance lab
            </p>
            <h1>
              Turn student data into <em>clear direction.</em>
            </h1>
            <p className="subtitle">
              Explore the factors behind a student&apos;s performance and get an
              instant, data-informed exam score estimate.
            </p>

            <div className="hero-meta" aria-label="Application highlights">
              <div className="hero-meta-item">
                <strong>19</strong>
                <span>profile inputs</span>
              </div>
              <div className="hero-meta-divider" />
              <div className="hero-meta-item">
                <strong>0.81</strong>
                <span>model R² score</span>
              </div>
              <div className="hero-meta-divider" />
              <div className="hero-meta-item">
                <strong>&lt; 1s</strong>
                <span>prediction time</span>
              </div>
            </div>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="visual-orbit visual-orbit-one" />
            <div className="visual-orbit visual-orbit-two" />
            <div className="visual-card">
              <div className="visual-card-top">
                <span className="visual-label">LIVE MODEL</span>
                <span className="visual-spark">✦</span>
              </div>
              <div className="visual-score">
                67<span>.1</span>
              </div>
              <p>Example predicted score</p>
              <div className="mini-bars">
                <span className="bar bar-one" />
                <span className="bar bar-two" />
                <span className="bar bar-three" />
                <span className="bar bar-four" />
                <span className="bar bar-five" />
                <span className="bar bar-six" />
                <span className="bar bar-seven" />
              </div>
              <div className="visual-card-footer">
                <span>{selectedModelInfo.label}</span>
                <span className="visual-footer-dot" />
                <span>Ready</span>
              </div>
            </div>
          </div>
        </section>

        <div className="dashboard">
          <section className="card form-card">
            <div className="card-topline">
              <div className="step-label">
                <span className="step-number">01</span>
                <span>Build a profile</span>
              </div>
              <button type="button" className="reset-button" onClick={resetForm}>
                <span aria-hidden="true">↻</span>
                Reset
              </button>
            </div>

            <div className="card-heading">
              <h2>Student profile</h2>
              <p>Adjust the inputs below to model a student&apos;s profile.</p>
            </div>

            <div className="model-selector">
              <div className="selector-header">
                <div>
                  <span className="selector-label">Prediction model</span>
                  <p>Choose the engine for this estimate</p>
                </div>
                <span className="selected-model-label">{selectedModelInfo.label}</span>
              </div>
              <div className="model-options" role="radiogroup" aria-label="Prediction model">
                {modelOptions.map((option) => {
                  const isSelected = selectedModel === option.value;

                  return (
                    <button
                      type="button"
                      className={`model-option ${isSelected ? "is-selected" : ""}`}
                      key={option.value}
                      role="radio"
                      aria-checked={isSelected}
                      onClick={() => setSelectedModel(option.value)}
                    >
                      <span className="model-option-icon" aria-hidden="true">
                        {option.label.slice(0, 1)}
                      </span>
                      <span className="model-option-copy">
                        <strong>{option.label}</strong>
                        <span>{option.description}</span>
                      </span>
                      <span className="model-option-check" aria-hidden="true">
                        {isSelected ? "✓" : ""}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="form-section">
              <div className="section-heading">
                <span className="section-icon" aria-hidden="true">↗</span>
                <div>
                  <h3>Study habits</h3>
                  <p>Daily routines and academic history</p>
                </div>
              </div>

              <div className="slider-grid">
                <SliderInput
                  id="hours-studied"
                  label="Hours studied"
                  value={hoursStudied}
                  min={0}
                  max={50}
                  suffix="h"
                  onChange={setHoursStudied}
                />
                <SliderInput
                  id="attendance"
                  label="Attendance"
                  value={attendance}
                  min={0}
                  max={100}
                  suffix="%"
                  onChange={setAttendance}
                />
                <SliderInput
                  id="previous-score"
                  label="Previous score"
                  value={previousScore}
                  min={0}
                  max={100}
                  onChange={setPreviousScore}
                />
                <SliderInput
                  id="sleep-hours"
                  label="Sleep hours"
                  value={sleepHours}
                  min={0}
                  max={12}
                  suffix="h"
                  onChange={setSleepHours}
                />
                <SliderInput
                  id="tutoring-sessions"
                  label="Tutoring sessions"
                  value={tutoringSessions}
                  min={0}
                  max={10}
                  onChange={setTutoringSessions}
                />
                <SliderInput
                  id="physical-activity"
                  label="Physical activity"
                  value={physicalActivity}
                  min={0}
                  max={20}
                  suffix="h"
                  onChange={setPhysicalActivity}
                />
              </div>
            </div>

            <div className="form-divider" />

            <div className="form-section">
              <div className="section-heading">
                <span className="section-icon section-icon-soft" aria-hidden="true">✦</span>
                <div>
                  <h3>Environment &amp; support</h3>
                  <p>Context that shapes the learning experience</p>
                </div>
              </div>

              <div className="select-grid">
                <SelectInput
                  id="parental-involvement"
                  label="Parental involvement"
                  value={parentalInvolvement}
                  onChange={setParentalInvolvement}
                  options={["Low", "Medium", "High"]}
                />
                <SelectInput
                  id="access-to-resources"
                  label="Access to resources"
                  value={accessToResources}
                  onChange={setAccessToResources}
                  options={["Low", "Medium", "High"]}
                />
                <SelectInput
                  id="motivation"
                  label="Motivation level"
                  value={motivation}
                  onChange={setMotivation}
                  options={["Low", "Medium", "High"]}
                />
                <SelectInput
                  id="family-income"
                  label="Family income"
                  value={familyIncome}
                  onChange={setFamilyIncome}
                  options={["Low", "Medium", "High"]}
                />
                <SelectInput
                  id="teacher-quality"
                  label="Teacher quality"
                  value={teacherQuality}
                  onChange={setTeacherQuality}
                  options={["Low", "Medium", "High"]}
                />
                <SelectInput
                  id="parental-education"
                  label="Parental education"
                  value={parentalEducation}
                  onChange={setParentalEducation}
                  options={["High School", "College", "Postgraduate"]}
                />
                <SelectInput
                  id="distance-from-home"
                  label="Distance from home"
                  value={distanceFromHome}
                  onChange={setDistanceFromHome}
                  options={["Near", "Moderate", "Far"]}
                />
                <SelectInput
                  id="peer-influence"
                  label="Peer influence"
                  value={peerInfluence}
                  onChange={setPeerInfluence}
                  options={["Negative", "Neutral", "Positive"]}
                />
                <SelectInput
                  id="school-type"
                  label="School type"
                  value={schoolType}
                  onChange={setSchoolType}
                  options={["Public", "Private"]}
                />
                <SelectInput
                  id="extracurricular"
                  label="Extracurricular activities"
                  value={extracurricular}
                  onChange={setExtracurricular}
                  options={["No", "Yes"]}
                />
                <SelectInput
                  id="internet-access"
                  label="Internet access"
                  value={internetAccess}
                  onChange={setInternetAccess}
                  options={["No", "Yes"]}
                />
                <SelectInput
                  id="learning-disabilities"
                  label="Learning disabilities"
                  value={learningDisabilities}
                  onChange={setLearningDisabilities}
                  options={["No", "Yes"]}
                />
                <SelectInput
                  id="gender"
                  label="Gender"
                  value={gender}
                  onChange={setGender}
                  options={["Male", "Female"]}
                />
              </div>
            </div>

            <div className="form-footer">
              {error && <p className="error-message">{error}</p>}
              <button
                type="button"
                className="predict-button"
                onClick={handlePredict}
                disabled={isLoading}
              >
                <span>{isLoading ? "Calculating profile" : "Generate prediction"}</span>
                <span className="button-arrow" aria-hidden="true">
                  {isLoading ? <span className="spinner" /> : "→"}
                </span>
              </button>
              <p className="form-note">Your inputs stay in this session and are not stored.</p>
            </div>
          </section>

          <aside className="sidebar">
            <section className="card prediction-card">
              <div className="card-topline">
                <div className="step-label">
                  <span className="step-number">02</span>
                  <span>Model output</span>
                </div>
                {predictedScore !== null && <span className="result-tag">New result</span>}
              </div>

              <div className="card-heading prediction-heading">
                <h2>Predicted score</h2>
                <p>Estimated exam performance out of 100.</p>
              </div>

              {predictedScore !== null ? (
                <div className="result-state">
                  <div
                    className="score-ring"
                    style={{
                      background: `conic-gradient(var(--accent) ${scorePercent}%, #dce8e6 0)`,
                    }}
                  >
                    <div className="score-ring-inner">
                      <strong>{predictedScore.toFixed(1)}</strong>
                      <span>/ 100</span>
                    </div>
                  </div>
                  <div className={`score-status ${scoreTone}`}>{scoreMessage}</div>
                  <div className="score-bar" aria-label={`Score: ${scorePercent}%`}>
                    <div className="score-fill" style={{ width: `${scorePercent}%` }} />
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon" aria-hidden="true">
                    <span>✦</span>
                  </div>
                  <strong>Your result will appear here</strong>
                  <p>Set the student profile and generate a prediction to get started.</p>
                </div>
              )}

              <div className="prediction-divider" />
              <div className="profile-snapshot">
                <div className="snapshot-heading">
                  <span>Current profile</span>
                  <span className="snapshot-status">Ready</span>
                </div>
                <div className="snapshot-grid">
                  {profileHighlights.map((item) => (
                    <div className="snapshot-item" key={item.label}>
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="card model-card">
              <div className="card-topline">
                <div className="step-label">
                  <span className="step-number">03</span>
                  <span>About the model</span>
                </div>
                <span className="model-badge">v1.0</span>
              </div>
              <div className="model-intro">
                <h2>{selectedModelInfo.label}</h2>
                <p>{selectedModelInfo.description} model trained on student performance factors.</p>
              </div>
              <div className="metrics">
                <div className="metric">
                  <span>Mean absolute error</span>
                  <strong>{selectedModelInfo.mae}</strong>
                </div>
                <div className="metric">
                  <span>R² score</span>
                  <strong>{selectedModelInfo.r2}</strong>
                </div>
              </div>
            </section>

            <div className="insight-card">
              <span className="insight-icon" aria-hidden="true">✦</span>
              <div>
                <strong>Use the score as a signal</strong>
                <p>Small changes in habits can shift the model&apos;s estimate.</p>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

interface SliderInputProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  suffix?: string;
  onChange: (value: number) => void;
}

function SliderInput({
  id,
  label,
  value,
  min,
  max,
  suffix = "",
  onChange,
}: SliderInputProps) {
  const progress = ((value - min) / (max - min)) * 100;

  return (
    <div className="input-group slider-group">
      <div className="field-label-row">
        <label htmlFor={id}>{label}</label>
        <span className="slider-value">{value}{suffix}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
        style={{
          background: `linear-gradient(90deg, var(--accent) ${progress}%, var(--track) ${progress}%)`,
        }}
      />
      <div className="range-limits" aria-hidden="true">
        <span>{min}{suffix}</span>
        <span>{max}{suffix}</span>
      </div>
    </div>
  );
}

interface SelectInputProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
}

function SelectInput({ id, label, value, onChange, options }: SelectInputProps) {
  return (
    <div className="input-group select-group">
      <label htmlFor={id}>{label}</label>
      <div className="select-wrap">
        <select id={id} value={value} onChange={(event) => onChange(event.target.value)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        <span className="select-chevron" aria-hidden="true">⌄</span>
      </div>
    </div>
  );
}

export default App;
