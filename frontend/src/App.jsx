import { useState } from 'react'

const initialForm = {
  gender: 'Female',
  senior_citizen: 0,
  partner: 'No',
  dependents: 'No',
  tenure: 12,
  phone_service: 'Yes',
  multiple_lines: 'No',
  internet_service: 'Fiber optic',
  online_security: 'No',
  online_backup: 'No',
  device_protection: 'No',
  tech_support: 'No',
  streaming_tv: 'No',
  streaming_movies: 'No',
  contract: 'Month-to-month',
  paperless_billing: 'Yes',
  payment_method: 'Electronic check',
  monthly_charges: 70.0,
  total_charges: 840.0,
}

const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function App() {
  const [form, setForm] = useState(initialForm)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...form,
          senior_citizen: Number(form.senior_citizen),
          tenure: Number(form.tenure),
          monthly_charges: Number(form.monthly_charges),
          total_charges: Number(form.total_charges),
        }),
      })
      if (!response.ok) throw new Error('Prediction failed')
      const data = await response.json()
      setResult({
        probability: data.churn_probability,
        label: data.churn_label,
        category: data.churn_probability >= 0.7 ? 'High' : data.churn_probability >= 0.4 ? 'Medium' : 'Low',
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="card">
        <h1>Telco Churn Predictor</h1>
        <p>Enter customer details to estimate churn probability.</p>
        <form onSubmit={handleSubmit}>
          <label>
            Gender
            <select name="gender" value={form.gender} onChange={handleChange}>
              <option>Female</option>
              <option>Male</option>
            </select>
          </label>
          <label>
            Senior Citizen (0/1)
            <input name="senior_citizen" type="number" min="0" max="1" value={form.senior_citizen} onChange={handleChange} />
          </label>
          <label>
            Partner
            <select name="partner" value={form.partner} onChange={handleChange}>
              <option>Yes</option>
              <option>No</option>
            </select>
          </label>
          <label>
            Dependents
            <select name="dependents" value={form.dependents} onChange={handleChange}>
              <option>Yes</option>
              <option>No</option>
            </select>
          </label>
          <label>
            Tenure (months)
            <input name="tenure" type="number" min="0" value={form.tenure} onChange={handleChange} />
          </label>
          <label>
            Phone Service
            <select name="phone_service" value={form.phone_service} onChange={handleChange}>
              <option>Yes</option>
              <option>No</option>
            </select>
          </label>
          <label>
            Multiple Lines
            <select name="multiple_lines" value={form.multiple_lines} onChange={handleChange}>
              <option>No</option>
              <option>Yes</option>
              <option>No phone service</option>
            </select>
          </label>
          <label>
            Internet Service
            <select name="internet_service" value={form.internet_service} onChange={handleChange}>
              <option>DSL</option>
              <option>Fiber optic</option>
              <option>No</option>
            </select>
          </label>
          <label>
            Online Security
            <select name="online_security" value={form.online_security} onChange={handleChange}>
              <option>No</option>
              <option>Yes</option>
              <option>No internet service</option>
            </select>
          </label>
          <label>
            Online Backup
            <select name="online_backup" value={form.online_backup} onChange={handleChange}>
              <option>No</option>
              <option>Yes</option>
              <option>No internet service</option>
            </select>
          </label>
          <label>
            Device Protection
            <select name="device_protection" value={form.device_protection} onChange={handleChange}>
              <option>No</option>
              <option>Yes</option>
              <option>No internet service</option>
            </select>
          </label>
          <label>
            Tech Support
            <select name="tech_support" value={form.tech_support} onChange={handleChange}>
              <option>No</option>
              <option>Yes</option>
              <option>No internet service</option>
            </select>
          </label>
          <label>
            Streaming TV
            <select name="streaming_tv" value={form.streaming_tv} onChange={handleChange}>
              <option>No</option>
              <option>Yes</option>
              <option>No internet service</option>
            </select>
          </label>
          <label>
            Streaming Movies
            <select name="streaming_movies" value={form.streaming_movies} onChange={handleChange}>
              <option>No</option>
              <option>Yes</option>
              <option>No internet service</option>
            </select>
          </label>
          <label>
            Contract
            <select name="contract" value={form.contract} onChange={handleChange}>
              <option>Month-to-month</option>
              <option>One year</option>
              <option>Two year</option>
            </select>
          </label>
          <label>
            Paperless Billing
            <select name="paperless_billing" value={form.paperless_billing} onChange={handleChange}>
              <option>Yes</option>
              <option>No</option>
            </select>
          </label>
          <label>
            Payment Method
            <select name="payment_method" value={form.payment_method} onChange={handleChange}>
              <option>Electronic check</option>
              <option>Mailed check</option>
              <option>Bank transfer (automatic)</option>
              <option>Credit card (automatic)</option>
            </select>
          </label>
          <label>
            Monthly Charges
            <input name="monthly_charges" type="number" step="0.01" value={form.monthly_charges} onChange={handleChange} />
          </label>
          <label>
            Total Charges
            <input name="total_charges" type="number" step="0.01" value={form.total_charges} onChange={handleChange} />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? 'Predicting…' : 'Predict churn'}
          </button>
        </form>
        {error && <div className="result">Error: {error}</div>}
        {result && (
          <div className="result">
            <p>Churn probability: {(result.probability * 100).toFixed(2)}%</p>
            <p>Risk category: {result.category}</p>
            <p>Predicted label: {result.label ? 'Will churn' : 'Will stay'}</p>
          </div>
        )}
      </div>
    </div>
  )
}
