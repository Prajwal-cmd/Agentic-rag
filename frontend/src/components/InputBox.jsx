import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Paperclip } from 'lucide-react';
import { useChatContext } from '../context/ChatContext';

const InputBox = ({ onSendMessage }) => {
  const [input, setInput] = useState('');
  const { isLoading } = useChatContext();
  const textareaRef = useRef(null);
  const maxLength = 2000;

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 150) + 'px';
    }
  }, [input]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-gradient-to-r from-white to-gray-50 dark:from-gray-800 dark:to-gray-900">
      <div className="flex gap-3 items-end max-w-5xl mx-auto">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask anything about your documents..."
            className="w-full px-4 py-3 pr-20 border-2 border-gray-300 dark:border-gray-600 rounded-2xl focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent outline-none resize-none bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 shadow-lg transition-all"
            rows={1}
            maxLength={maxLength}
            disabled={isLoading}
            style={{
              minHeight: '56px',
              maxHeight: '150px',
            }}
          />
          <div className="absolute bottom-3 right-3 flex items-center gap-2">
            <span className="text-xs text-gray-400 dark:text-gray-500 font-medium">
              {input.length}/{maxLength}
            </span>
          </div>
        </div>

        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-6 py-3.5 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 disabled:from-gray-400 disabled:to-gray-400 text-white rounded-2xl font-semibold transition-all disabled:cursor-not-allowed shadow-lg hover:shadow-xl flex items-center gap-2 flex-shrink-0 h-[56px]"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="hidden sm:inline">Processing...</span>
            </>
          ) : (
            <>
              <Send className="w-5 h-5" />
              <span className="hidden sm:inline">Send</span>
            </>
          )}
        </button>
      </div>

      <div className="text-xs text-gray-500 dark:text-gray-400 text-center mt-2 space-x-4">
        <span>Press <kbd className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded font-mono">Enter</kbd> to send</span>
        <span>•</span>
        <span><kbd className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded font-mono">Shift+Enter</kbd> for new line</span>
      </div>
    </form>
  );
};

export default InputBox;
