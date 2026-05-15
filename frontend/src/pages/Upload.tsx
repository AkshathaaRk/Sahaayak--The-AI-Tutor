import React, { useState } from 'react';
import { Upload as UploadIcon, FileText, Image, Video, X, FileIcon, AlertCircle } from 'lucide-react';
import Header from '../components/Layout/Header';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import { useApp } from '../contexts/AppContext';
import { uploadAPI, FileAnalysis, healthAPI } from '../services/api';

const Upload: React.FC = () => {
  const { loading, setLoading, user } = useApp();
  const [files, setFiles] = useState<Array<{ file: File; name: string; size: number; type: string; id: string }>>([]);
  const [outputOptions, setOutputOptions] = useState({
    summarize: true,
    keyPoints: false,
    quiz: false,
    studyGuide: false
  });
  const [analysisResults, setAnalysisResults] = useState<FileAnalysis[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Test function to check backend connection
  const testConnection = async () => {
    console.log('🧪 Testing backend connection...');
    try {
      const response = await healthAPI.check();
      if (response.success) {
        alert('✅ Backend connection successful!');
        console.log('✅ Health check passed:', response.data);
      } else {
        alert('❌ Backend health check failed: ' + response.error);
        console.error('❌ Health check failed:', response.error);
      }
    } catch (err) {
      alert('💥 Connection test failed: ' + err);
      console.error('💥 Connection test error:', err);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = Array.from(event.target.files || []);
    const newFiles = uploadedFiles.map(file => ({
      file: file,
      name: file.name,
      size: file.size,
      type: file.type,
      id: Math.random().toString(36).substring(2, 11)
    }));
    setFiles(prev => [...prev, ...newFiles]);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    const newFiles = droppedFiles.map(file => ({
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      id: Math.random().toString(36).substring(2, 11)
    }));
    setFiles(prev => [...prev, ...newFiles]);
  };

  const triggerFileInput = () => {
    const fileInput = document.getElementById('file-upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.click();
    }
  };

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(file => file.id !== id));
  };

  const handleProcess = async () => {
    if (files.length === 0) {
      setError('Please upload at least one file');
      return;
    }

    setIsProcessing(true);
    setLoading(true);
    setError(null);
    setAnalysisResults([]);

    try {
      const results: FileAnalysis[] = [];

      // Process each file
      for (const fileItem of files) {
        const response = await uploadAPI.analyze(fileItem.file, {
          analysis_type: 'educational',
          question: 'Analyze this file for educational content and create a summary',
          language: 'en',
          user_id: user?.id || 'sahayak_user',
          session_id: `upload_${Date.now()}`
        });

        if (response.success && response.data) {
          results.push(response.data.analysis);
        } else {
          console.error(`Failed to analyze ${fileItem.name}:`, response.error);
        }
      }

      if (results.length > 0) {
        setAnalysisResults(results);
        setError(null);
      } else {
        setError('Failed to analyze any files. Please try again.');
      }
    } catch (err) {
      setError('Network error. Please check if the AI backend is running.');
    } finally {
      setIsProcessing(false);
      setLoading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (type: string) => {
    if (type.includes('image')) return <Image className="w-5 h-5" />;
    if (type.includes('video')) return <Video className="w-5 h-5" />;
    if (type.includes('pdf') || type.includes('document')) return <FileText className="w-5 h-5" />;
    return <FileIcon className="w-5 h-5" />;
  };

  return (
    <div className="min-h-screen bg-grey-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-grey-800 mb-2">Generate Study Material from Files</h1>
          <p className="text-grey-600">
            Upload your documents, presentations, images, or videos to create student-friendly study materials.
          </p>
          {/* Temporary test button */}
          <div className="mt-4">
            <Button variant="outline" size="sm" onClick={testConnection}>
              🧪 Test Backend Connection
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <Card>
            <h2 className="text-xl font-semibold text-grey-800 mb-6">Upload Files</h2>
            
            {/* File Upload Area */}
            <div className="space-y-6">
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  isDragOver
                    ? 'border-lavender-500 bg-lavender-50'
                    : 'border-grey-300 hover:border-lavender-500'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <UploadIcon className="w-12 h-12 text-grey-400 mx-auto mb-4" />
                <p className="text-grey-600 mb-4">Drag & drop files here or click to browse</p>
                <input
                  type="file"
                  multiple
                  accept=".pdf,.doc,.docx,.ppt,.pptx,.jpg,.jpeg,.png,.mp4,.mov"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <div className="space-x-3">
                  <label htmlFor="file-upload" className="inline-block cursor-pointer">
                    <div className="px-4 py-2 border border-grey-300 rounded-lg bg-white hover:bg-grey-50 transition-colors">
                      Select Files
                    </div>
                  </label>
                  <button
                    onClick={triggerFileInput}
                    className="px-4 py-2 bg-lavender-500 text-white rounded-lg hover:bg-lavender-600 transition-colors"
                  >
                    Browse Files
                  </button>
                </div>
                <p className="text-sm text-grey-500 mt-2">
                  Supports PDF, DOC, PPT, Images, Videos (Max 10MB per file)
                </p>
              </div>

              {/* File List */}
              {files.length > 0 && (
                <div className="space-y-3">
                  <h3 className="font-medium text-grey-800">Uploaded Files</h3>
                  {files.map((file) => (
                    <div key={file.id} className="flex items-center justify-between p-3 bg-grey-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        {getFileIcon(file.type)}
                        <div>
                          <p className="text-sm font-medium text-grey-800">{file.name}</p>
                          <p className="text-xs text-grey-600">{formatFileSize(file.size)}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeFile(file.id)}
                        className="text-grey-400 hover:text-red-500 transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Output Options */}
              <div className="space-y-3">
                <h3 className="font-medium text-grey-800">What would you like to generate?</h3>
                <div className="space-y-2">
                  {Object.entries(outputOptions).map(([key, value]) => (
                    <label key={key} className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={value}
                        onChange={(e) => setOutputOptions(prev => ({ ...prev, [key]: e.target.checked }))}
                        className="w-4 h-4 text-lavender-600 border-grey-300 rounded focus:ring-lavender-500"
                      />
                      <span className="text-sm text-grey-700">
                        {key === 'summarize' && 'Create Summary'}
                        {key === 'keyPoints' && 'Extract Key Points'}
                        {key === 'quiz' && 'Generate Quiz Questions'}
                        {key === 'studyGuide' && 'Create Study Guide'}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              <Button
                variant="primary"
                size="lg"
                className="w-full"
                onClick={handleProcess}
                loading={isProcessing}
                icon={UploadIcon}
              >
                Process & Generate
              </Button>
            </div>
          </Card>

          {/* Generated Content */}
          <Card>
            <h2 className="text-xl font-semibold text-grey-800 mb-6">Generated Study Material</h2>

            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-red-800 font-medium">Error</p>
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              </div>
            )}

            {isProcessing ? (
              <div className="flex flex-col items-center justify-center py-12">
                <LoadingSpinner size="lg" />
                <p className="text-grey-600 mt-4">Processing your files...</p>
                <div className="w-full bg-grey-200 rounded-full h-2 mt-4">
                  <div className="bg-lavender-500 h-2 rounded-full animate-pulse" style={{ width: '65%' }}></div>
                </div>
              </div>
            ) : analysisResults.length > 0 ? (
              <div className="space-y-6">
                {analysisResults.map((result) => (
                  <div key={result.id} className="bg-grey-50 p-6 rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-medium text-grey-800">{result.filename}</h3>
                      <span className={`text-xs px-2 py-1 rounded ${
                        result.type === 'pdf' ? 'bg-red-100 text-red-700' :
                        result.type === 'image' ? 'bg-green-100 text-green-700' :
                        result.type === 'text' ? 'bg-blue-100 text-blue-700' :
                        'bg-gray-100 text-gray-700'
                      }`}>
                        {result.type.toUpperCase()}
                      </span>
                    </div>

                    {result.summary && (
                      <div className="mb-4">
                        <h4 className="font-medium text-grey-700 mb-2">📄 Summary:</h4>
                        <p className="text-sm text-grey-600 leading-relaxed">{result.summary}</p>
                      </div>
                    )}

                    {result.analysis && (
                      <div className="mb-4">
                        <h4 className="font-medium text-grey-700 mb-2">🔍 Analysis:</h4>
                        <p className="text-sm text-grey-600 leading-relaxed">{result.analysis}</p>
                      </div>
                    )}

                    {result.model_used && (
                      <div className="text-xs text-grey-500 mt-4">
                        🤖 Analyzed by: {result.model_used}
                      </div>
                    )}
                  </div>
                ))}

                <div className="flex flex-wrap gap-3">
                  <Button variant="secondary">Save All</Button>
                  <Button variant="outline">Edit</Button>
                  <Button variant="outline">Download</Button>
                </div>

                <div className="border-t pt-6">
                  <p className="text-sm text-grey-600 mb-3">Was this content helpful?</p>
                  <div className="flex items-center space-x-4">
                    <Button variant="ghost" size="sm">Yes</Button>
                    <Button variant="ghost" size="sm">No</Button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-grey-500">
                <FileText className="w-12 h-12 mb-4" />
                <p>Upload files and click "Process & Generate" to create study materials!</p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Upload;