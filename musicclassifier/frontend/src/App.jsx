import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  
  const [explanation, setExplanation] = useState(null);
  const [loadingExplain, setLoadingExplain] = useState(false);
  
  const [chatHistory, setChatHistory] = useState([]);
  const [chatQuestion, setChatQuestion] = useState("");
  const [loadingChat, setLoadingChat] = useState(false);
  
  const [recommendations, setRecommendations] = useState(null);
  const [loadingRecommend, setLoadingRecommend] = useState(false);
  
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, loadingChat]);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
    setExplanation(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(URL.createObjectURL(selectedFile));
  };

  const handleAnalyze = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    setResult(null);
    setExplanation(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      const data = response.data;
      setResult(data);
      
      setHistory(prev => [
        {
          filename: file.name,
          genre: data.genre,
          probabilities: data.probabilities,
          timestamp: new Date().toLocaleString()
        },
        ...prev
      ].slice(0, 5));
      
    } catch (err) {
      console.error(err);
      setError('Ocurrió un error al conectar con el servidor. Verifica que el backend esté en ejecución.');
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    if (!result) return;
    setLoadingExplain(true);
    try {
      const response = await axios.post('http://localhost:8000/explain', {
        genre: result.genre,
        probabilities: result.probabilities
      });
      setExplanation(response.data.explanation);
    } catch (err) {
      console.error(err);
      setExplanation("Error al obtener la explicación de la IA.");
    } finally {
      setLoadingExplain(false);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatQuestion.trim()) return;

    const newHistory = [...chatHistory, { role: 'user', content: chatQuestion }];
    setChatHistory(newHistory);
    setChatQuestion("");
    setLoadingChat(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        question: chatQuestion,
        history: chatHistory
      });
      setChatHistory([...newHistory, { role: 'assistant', content: response.data.response }]);
    } catch (err) {
      console.error(err);
      setChatHistory([...newHistory, { role: 'assistant', content: "Lo siento, ocurrió un error al procesar tu mensaje." }]);
    } finally {
      setLoadingChat(false);
    }
  };

  const handleRecommend = async () => {
    if (history.length < 2) return;
    setLoadingRecommend(true);
    try {
      const response = await axios.post('http://localhost:8000/recommend', {
        history: history
      });
      setRecommendations(response.data.recommendations);
    } catch (err) {
      console.error(err);
      setRecommendations("Error al obtener las recomendaciones.");
    } finally {
      setLoadingRecommend(false);
    }
  };

  const formatDataForChart = (probabilities) => {
    if (!probabilities) return [];
    return Object.entries(probabilities)
      .map(([name, value]) => ({ name, value: value * 100 }))
      .sort((a, b) => b.value - a.value);
  };

  const chartData = result ? formatDataForChart(result.probabilities) : [];

  return (
    <div className="min-h-screen bg-[#0f0f13] text-white font-sans p-4 md:p-8">
      <header className="max-w-4xl mx-auto mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-bold mb-2 flex items-center justify-center gap-3">
          <span className="text-[#7c3aed]">🎵</span>
          MusicClassifier
        </h1>
        <p className="text-gray-400 text-lg">Clasificador de géneros musicales con IA</p>
      </header>

      <main className="max-w-4xl mx-auto space-y-8">
        <section className="bg-[#1a1a2e] border border-[#2d2d44] rounded-2xl p-6 md:p-10 shadow-lg">
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="border-2 border-dashed border-[#2d2d44] rounded-xl p-10 flex flex-col items-center justify-center text-center transition-colors hover:border-[#7c3aed] cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              accept=".mp3,.wav"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            <div className="text-4xl mb-4">📁</div>
            <p className="text-xl font-semibold mb-2">Arrastra y suelta tu archivo de audio aquí</p>
            <p className="text-gray-400 mb-6">o haz clic para seleccionar (MP3 o WAV)</p>
            <button className="bg-[#2d2d44] hover:bg-[#7c3aed] text-white font-medium py-2 px-6 rounded-full transition-colors">
              Seleccionar archivo
            </button>
          </div>

          {file && (
            <div className="mt-8 flex flex-col items-center">
              <p className="text-lg font-medium text-[#06b6d4] mb-4">🎵 {file.name}</p>
              {audioUrl && (
                <audio controls src={audioUrl} className="w-full max-w-md mb-6" />
              )}
              
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className={`w-full max-w-md py-3 px-6 rounded-xl font-bold text-lg transition-all ${
                  loading 
                    ? 'bg-gray-600 cursor-not-allowed' 
                    : 'bg-[#7c3aed] hover:bg-[#6d28d9] shadow-lg shadow-[#7c3aed]/20'
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analizando...
                  </span>
                ) : (
                  'Analizar Género'
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-6 bg-red-500/20 border border-red-500 text-red-200 px-4 py-3 rounded-xl text-center">
              {error}
            </div>
          )}
        </section>

        {result && (
          <section className="bg-[#1a1a2e] border border-[#2d2d44] rounded-2xl p-6 md:p-10 shadow-lg">
            <h2 className="text-2xl font-bold mb-6 text-center">Resultado de Predicción</h2>
            
            <div className="text-center mb-10">
              <p className="text-gray-400 uppercase tracking-widest text-sm font-semibold mb-2">Género Predicho</p>
              <div className="inline-block bg-[#2d2d44]/50 border border-[#7c3aed]/30 rounded-2xl px-8 py-4">
                <span className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-[#7c3aed] to-[#06b6d4]">
                  {result.genre}
                </span>
              </div>
            </div>

            <div className="h-80 w-full mb-8">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                  <XAxis type="number" domain={[0, 100]} stroke="#6b7280" />
                  <YAxis dataKey="name" type="category" stroke="#e5e7eb" width={100} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1a2e', borderColor: '#2d2d44', borderRadius: '0.5rem' }}
                    itemStyle={{ color: '#fff' }}
                    formatter={(value) => [`${value.toFixed(1)}%`, 'Probabilidad']}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.name === result.genre ? '#7c3aed' : '#2d2d44'} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-8 border-t border-[#2d2d44] pt-8 text-center">
              {!explanation && !loadingExplain && (
                <button
                  onClick={handleExplain}
                  className="bg-[#06b6d4]/20 hover:bg-[#06b6d4]/30 text-[#06b6d4] border border-[#06b6d4]/50 font-bold py-2 px-6 rounded-full transition-colors"
                >
                  ✨ Explicar predicción
                </button>
              )}
              
              {loadingExplain && (
                <div className="flex justify-center items-center gap-2 text-[#06b6d4]">
                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Consultando IA...</span>
                </div>
              )}

              {explanation && (
                <div className="text-left bg-[#0f0f13] border border-[#06b6d4]/50 rounded-xl p-6 mt-4">
                  <h3 className="text-[#06b6d4] font-bold text-lg mb-2 flex items-center gap-2">
                    ✨ ¿Por qué este género?
                  </h3>
                  <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">{explanation}</p>
                </div>
              )}
            </div>
          </section>
        )}

        {history.length > 0 && (
          <section className="bg-[#1a1a2e] border border-[#2d2d44] rounded-2xl p-6 shadow-lg">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-xl font-bold">Historial de Análisis</h3>
              <div className="flex gap-2">
                {history.length >= 2 && (
                  <button 
                    onClick={handleRecommend}
                    disabled={loadingRecommend}
                    className="text-sm bg-[#7c3aed]/20 text-[#7c3aed] hover:bg-[#7c3aed]/30 transition-colors px-3 py-1 rounded-lg border border-[#7c3aed]/30 disabled:opacity-50"
                  >
                    {loadingRecommend ? 'Cargando...' : '🎯 Recomendar'}
                  </button>
                )}
                <button 
                  onClick={() => {
                    setHistory([]);
                    setRecommendations(null);
                  }}
                  className="text-sm text-gray-400 hover:text-white transition-colors px-3 py-1 bg-[#2d2d44] rounded-lg"
                >
                  Limpiar
                </button>
              </div>
            </div>
            
            <div className="space-y-3 mb-6">
              {history.map((item, idx) => (
                <div key={idx} className="flex flex-col sm:flex-row sm:items-center justify-between p-4 bg-[#0f0f13] rounded-xl border border-[#2d2d44]/50">
                  <div className="mb-2 sm:mb-0">
                    <p className="font-medium truncate max-w-xs">{item.filename}</p>
                    <p className="text-xs text-gray-500">{item.timestamp}</p>
                  </div>
                  <div className="bg-[#7c3aed]/20 text-[#7c3aed] px-4 py-1.5 rounded-full font-bold text-sm text-center border border-[#7c3aed]/30">
                    {item.genre}
                  </div>
                </div>
              ))}
            </div>

            {recommendations && (
              <div className="text-left bg-[#0f0f13] border border-[#7c3aed]/50 rounded-xl p-6 mt-4">
                <h3 className="text-[#7c3aed] font-bold text-lg mb-2 flex items-center gap-2">
                  🎯 Recomendaciones Musicales
                </h3>
                <p className="text-gray-300 leading-relaxed whitespace-pre-wrap">{recommendations}</p>
              </div>
            )}
          </section>
        )}

        <section className="bg-[#1a1a2e] border border-[#2d2d44] rounded-2xl p-6 shadow-lg flex flex-col h-[500px]">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">💬 Chat Musical</h3>
          
          <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2 custom-scrollbar">
            {chatHistory.length === 0 ? (
              <div className="text-center text-gray-500 h-full flex items-center justify-center">
                Hazme cualquier pregunta sobre géneros musicales, artistas o historia de la música.
              </div>
            ) : (
              chatHistory.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div 
                    className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                      msg.role === 'user' 
                        ? 'bg-[#7c3aed] text-white rounded-tr-sm' 
                        : 'bg-[#2d2d44] text-gray-200 border border-[#2d2d44] rounded-tl-sm'
                    }`}
                  >
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  </div>
                </div>
              ))
            )}
            {loadingChat && (
              <div className="flex justify-start">
                <div className="bg-[#2d2d44] text-gray-400 rounded-2xl rounded-tl-sm px-5 py-3 border border-[#2d2d44] flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  🎵 Pensando...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <form onSubmit={handleChatSubmit} className="flex gap-2">
            <input
              type="text"
              value={chatQuestion}
              onChange={(e) => setChatQuestion(e.target.value)}
              placeholder="Pregunta sobre géneros musicales..."
              className="flex-1 bg-[#0f0f13] border border-[#2d2d44] rounded-xl px-4 py-3 focus:outline-none focus:border-[#7c3aed] transition-colors"
              disabled={loadingChat}
            />
            <button
              type="submit"
              disabled={loadingChat || !chatQuestion.trim()}
              className="bg-[#7c3aed] hover:bg-[#6d28d9] disabled:bg-[#2d2d44] disabled:text-gray-500 font-bold py-3 px-6 rounded-xl transition-colors"
            >
              Enviar
            </button>
          </form>
        </section>

      </main>
    </div>
  );
}

export default App;
