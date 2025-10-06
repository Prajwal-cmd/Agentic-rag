import React, { useState } from 'react';
import { ChatProvider } from './context/ChatContext';
import ChatInterface from './components/ChatInterface';
import ResearchFeatures from './components/ResearchFeatures';

function App() {
  const [currentView, setCurrentView] = useState('chat'); // 'chat' or 'research'
  const [sessionId] = useState(() => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);

  return (
    <ChatProvider>
      <div className="min-h-screen bg-gray-50">
        {/* Header with Navigation */}
        <header className="bg-gradient-to-r from-indigo-600 to-purple-600 shadow-lg">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                🎓 Research Assistant Pro
              </h1>
              <nav className="flex gap-3">
                <button
                  onClick={() => setCurrentView('chat')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'chat'
                      ? 'bg-white text-indigo-600 shadow-md'
                      : 'bg-indigo-500 text-white hover:bg-indigo-400'
                  }`}
                >
                  💬 Chat
                </button>
                <button
                  onClick={() => setCurrentView('research')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'research'
                      ? 'bg-white text-indigo-600 shadow-md'
                      : 'bg-indigo-500 text-white hover:bg-indigo-400'
                  }`}
                >
                  🔬 Research Tools
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="py-8">
          {currentView === 'chat' ? (
            <ChatInterface />
          ) : (
            <ResearchFeatures sessionId={sessionId} />
          )}
        </main>
      </div>
    </ChatProvider>
  );
}

export default App;
