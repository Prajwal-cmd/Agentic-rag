import React, { useState } from 'react';
import { Send } from 'lucide-react';
import { useChatContext } from '../context/ChatContext';

const InputBox = ({ onSendMessage }) => {
  const [input, setInput] = useState('');
  const { isLoading } = useChatContext();
  const maxLength = 2000;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t border-gray-200 p-4 bg-white">
      <form onSubmit={handleSubmit} className="flex items-end space-x-3">
        <div className="flex-1">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value.slice(0, maxLength))}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isLoading}
            rows={3}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none disabled:bg-gray-100 disabled:text-gray-500"
          />
          <div className="flex justify-between mt-2 px-1">
            <span className="text-xs text-gray-500">
              Shift + Enter for new line
            </span>
            <span className={`text-xs ${input.length > maxLength * 0.9 ? 'text-red-500 font-medium' : 'text-gray-500'}`}>
              {input.length}/{maxLength}
            </span>
          </div>
        </div>
        
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center h-[52px] transition-colors"
        >
          {isLoading ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </form>
    </div>
  );
};

export default InputBox;