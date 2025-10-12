import React, { useState, useRef } from 'react';
import { Upload, X, File, AlertCircle, CheckCircle, FileCheck } from 'lucide-react';
import { useChatContext } from '../context/ChatContext';
import { chatAPI } from '../api/apiClient';
import { formatFileSize, isValidFileType } from '../utils/formatters';

const DocumentUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const fileInputRef = useRef(null);

  const { sessionId, setDocumentsUploaded, addMessage, addUploadedDocument, uploadedDocuments } = useChatContext();
  const maxSize = 15 * 1024 * 1024; // 15MB

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    handleFiles(files);
  };

  const handleFiles = (files) => {
    setUploadError(null);
    setUploadSuccess(null);

    // Validate file types
    const invalidFiles = files.filter((f) => !isValidFileType(f.name));
    if (invalidFiles.length > 0) {
      setUploadError(`Invalid file types: ${invalidFiles.map((f) => f.name).join(', ')}`);
      return;
    }

    // Validate total size
    const totalSize = files.reduce((sum, f) => sum + f.size, 0);
    if (totalSize > maxSize) {
      setUploadError(`Total file size exceeds 15MB limit (${formatFileSize(totalSize)})`);
      return;
    }

    setSelectedFiles(files);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const removeFile = (index) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setUploadError(null);
    setUploadSuccess(null);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0 || !sessionId) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      console.log('Uploading with session ID:', sessionId);
      const result = await chatAPI.uploadDocuments(selectedFiles, sessionId);
      console.log('Upload result:', result);

      // Verify session IDs match
      if (result.session_id !== sessionId) {
        console.error(`Session ID mismatch: sent ${sessionId}, got ${result.session_id}`);
        setUploadError('Session ID mismatch error. Please refresh the page.');
        return;
      }

      const successMsg = `Successfully uploaded ${result.files_processed} file(s), created ${result.chunks_created} chunks`;
      setUploadSuccess(successMsg);
      addMessage('system', `✓ ${successMsg}`);
      setDocumentsUploaded(true);

      // NEW: Add each uploaded document to the list
      selectedFiles.forEach((file) => {
        addUploadedDocument(file.name, result.chunks_created);
      });

      setSelectedFiles([]);

      // Clear success message after 5 seconds
      setTimeout(() => setUploadSuccess(null), 5000);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Upload failed';
      setUploadError(errorMsg);
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Upload Zone */}
      <div
        className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl p-6 text-center hover:border-blue-500 dark:hover:border-blue-400 transition-colors cursor-pointer bg-gray-50 dark:bg-gray-800/50"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          onChange={handleFileSelect}
          className="hidden"
        />

        <Upload className="w-12 h-12 mx-auto mb-3 text-gray-400 dark:text-gray-500" />
        <p className="text-gray-700 dark:text-gray-300 font-medium mb-1">
          {uploading ? 'Uploading...' : 'Click or drag files here'}
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          PDF, DOCX, TXT (max 15MB total)
        </p>
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
            Selected files ({selectedFiles.length}):
          </p>
          <div className="space-y-2 mb-4">
            {selectedFiles.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <File className="w-4 h-4 text-blue-600 dark:text-blue-400 flex-shrink-0" />
                  <span className="text-sm text-gray-700 dark:text-gray-300 truncate">
                    {file.name}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400 flex-shrink-0">
                    {formatFileSize(file.size)}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(idx);
                  }}
                  className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded flex-shrink-0"
                >
                  <X className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                </button>
              </div>
            ))}
          </div>

          <button
            onClick={(e) => {
              e.stopPropagation();
              handleUpload();
            }}
            disabled={uploading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white py-2 px-4 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
          >
            {uploading ? 'Uploading...' : 'Upload Files'}
          </button>
        </div>
      )}

      {/* Success Message */}
      {uploadSuccess && (
        <div className="flex items-start gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-green-700 dark:text-green-300">{uploadSuccess}</p>
        </div>
      )}

      {/* Error Message */}
      {uploadError && (
        <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700 dark:text-red-300">{uploadError}</p>
        </div>
      )}

      {/* NEW: Uploaded Documents List */}
      {uploadedDocuments.length > 0 && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/10 dark:to-purple-900/10 rounded-lg border border-blue-200 dark:border-blue-800 p-4">
          <div className="flex items-center gap-2 mb-3">
            <FileCheck className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <h3 className="font-semibold text-gray-800 dark:text-gray-200">
              Uploaded Documents ({uploadedDocuments.length})
            </h3>
          </div>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {uploadedDocuments.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center justify-between p-2 bg-white/60 dark:bg-gray-800/60 rounded-lg backdrop-blur-sm"
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <div className="w-2 h-2 bg-green-500 rounded-full flex-shrink-0 animate-pulse"></div>
                  <span className="text-sm text-gray-700 dark:text-gray-300 truncate font-medium">
                    {doc.filename}
                  </span>
                </div>
                <span className="text-xs text-gray-500 dark:text-gray-400 flex-shrink-0">
                  {doc.chunksCount} chunks
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info */}
      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
        <p>
          <strong className="text-gray-700 dark:text-gray-300">Supported formats:</strong> PDF,
          DOCX, TXT
        </p>
        <p>
          <strong className="text-gray-700 dark:text-gray-300">Max size:</strong> 15MB total
        </p>
      </div>
    </div>
  );
};

export default DocumentUpload;
