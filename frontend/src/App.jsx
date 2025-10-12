import React, { useState } from 'react';
import { ChatProvider } from './context/ChatContext';
import ChatInterface from './components/ChatInterface';
import ResearchFeatures from './components/ResearchFeatures';
import { MessageSquare, Microscope } from 'lucide-react';

function App() {
  const [currentView, setCurrentView] = useState('chat');
  const [sessionId] = useState(() => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);

  return (
    <ChatProvider>
      <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900">
        {/* Enhanced Header with Navigation */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
          <div className="px-6 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Agentic RAG System
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Intelligent Document Analysis & Research Assistant
                </p>
              </div>

              {/* Navigation Tabs */}
              <nav className="flex gap-2">
                <button
                  onClick={() => setCurrentView('chat')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'chat'
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  <MessageSquare className="w-5 h-5" />
                  <span>Chat</span>
                </button>

                <button
                  onClick={() => setCurrentView('research')}
                  className={`relative flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'research'
                      ? 'bg-purple-600 text-white shadow-md'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  <Microscope className="w-5 h-5" />
                  <span>Research Tools</span>
                  {/* NEW Badge */}
                  <span className="absolute -top-1 -right-1 bg-gradient-to-r from-pink-500 to-red-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full shadow-lg animate-pulse">
                    NEW
                  </span>
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content - Takes remaining height */}
        <main className="flex-1 overflow-hidden">
          {currentView === 'chat' ? (
            <ChatInterface />
          ) : (
            <div className="h-full overflow-y-auto">
              <ResearchFeatures sessionId={sessionId} />
            </div>
          )}
        </main>
      </div>
    </ChatProvider>
  );
}

export default App;
