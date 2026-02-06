'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Settings, Key, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChatMessage, Message } from '@/components/ChatMessage';
import { v4 as uuidv4 } from 'uuid';

export default function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const [apiKey, setApiKey] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stored = localStorage.getItem('gemini_api_key');
    if (stored) setApiKey(stored);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleKeyChange = (val: string) => {
    setApiKey(val);
    localStorage.setItem('gemini_api_key', val);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    setIsLoading(true);

    const userMsg: Message = {
      id: uuidv4(),
      role: 'user',
      content: userQuery
    };
    
    const botMsgId = uuidv4();
    const botPlaceholder: Message = {
      id: botMsgId,
      role: 'assistant',
      isLoading: true
    };

    setMessages(prev => [...prev, userMsg, botPlaceholder]);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://mangonnn-stock-agent.hf.space';
      
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-gemini-api-key': apiKey
        },
        body: JSON.stringify({ 
          query: userQuery,
          timeframe_days: 7 
        }),
      });

      if (!response.ok) {
         if (response.status === 429) throw new Error("API Limit Reached. Please try again later.");
         throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const contentType = response.headers.get("content-type");
      
      if (contentType && contentType.includes("application/json")) {
        const result = await response.json();
        
        setMessages(prev => prev.map(msg =>
          msg.id === botMsgId
            ? { ...msg, isLoading: false, data: result, content: result.final_report }
            : msg
        ));
        
      } else {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        
        if (!reader) throw new Error("No response body");
  
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
  
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n\n');
  
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6);
              if (!jsonStr.trim()) continue;
  
              try {
                const data = JSON.parse(jsonStr);
                setMessages(prev => prev.map(msg => {
                  if (msg.id !== botMsgId) return msg;
  
                  if (data.error) {
                    return { ...msg, isLoading: false, isError: true, content: data.error };
                  }
  
                  if (data.type === 'progress') {
                    return msg;
                  }
  
                  if (data.type === 'result') {
                    return {
                      ...msg,
                      isLoading: false,
                      data: data,
                      content: data.final_report
                    };
                  }
  
                  return msg;
                }));
  
              } catch (e) {
                console.error("Error parsing stream chunk", e);
              }
            }
          }
        }
      }

    } catch (err: any) {
      console.error(err);
      setMessages(prev => prev.map(msg => 
        msg.id === botMsgId 
          ? { ...msg, isLoading: false, isError: true, content: err.message || "Something went wrong." } 
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex flex-col h-screen bg-black text-white font-mono text-sm overflow-hidden">
      
      {/* Header */}
      <header className="flex-none z-10 border-b border-gray-900 bg-black p-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
             <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
             <h1 className="text-sm font-bold uppercase tracking-widest text-gray-300">Stock Agent // <span className="text-gray-600">v1.0</span></h1>
          </div>
          <button 
            onClick={() => setShowSettings(!showSettings)}
            className="text-gray-500 hover:text-white transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="flex-none z-20 bg-gray-900 border-b border-gray-800"
          >
            <div className="max-w-5xl mx-auto p-4">
                <div className="flex items-center space-x-2 text-sm font-medium text-gray-300 mb-2">
                  <Key className="w-4 h-4 text-yellow-500" />
                  <span>Gemini API Key</span>
                </div>
                <input 
                  type="password" 
                  value={apiKey}
                  onChange={(e) => handleKeyChange(e.target.value)}
                  placeholder="AIzaSy..."
                  className="w-full max-w-md bg-black/50 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:ring-1 focus:ring-blue-500 outline-none"
                />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto z-0 p-4 scroll-smooth">
        <div className="max-w-4xl mx-auto min-h-full flex flex-col justify-end">
           {messages.length === 0 ? (
             <div className="flex flex-col items-center justify-center h-full text-center space-y-4 py-20 opacity-30">
                <div className="w-16 h-16 border border-gray-800 rounded-full flex items-center justify-center">
                   <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div className="space-y-1">
                   <h2 className="text-sm font-mono tracking-widest uppercase">System Ready</h2>
                   <p className="text-xs text-gray-500 font-mono">Awaiting Input...</p>
                </div>
             </div>
           ) : (
             messages.map(msg => (
               <ChatMessage key={msg.id} message={msg} />
             ))
           )}
           <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-none z-10 p-4 bg-black border-t border-gray-900">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative flex items-center bg-black border border-gray-800 rounded-lg p-1 focus-within:border-white transition-colors">
            <input 
              type="text" 
              placeholder="COMMAND >>" 
              className="w-full bg-transparent border-none focus:ring-0 outline-none text-white placeholder-gray-700 px-4 py-3 font-mono text-xs"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
            />
            <button 
              type="submit"
              disabled={!input.trim() || isLoading}
              className="bg-white hover:bg-gray-200 text-black p-2 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>

    </main>
  );
}
