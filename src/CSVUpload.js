import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Typography,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  Chip
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import AddIcon from '@mui/icons-material/Add';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';
import PredictIcon from '@mui/icons-material/Psychology';
import DownloadIcon from '@mui/icons-material/FileDownload';
import * as XLSX from 'xlsx';
import Papa from 'papaparse';
import axios from 'axios';

const CSVUpload = () => {  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [headers, setHeaders] = useState([]);
  const [data, setData] = useState([]);
  const [allData, setAllData] = useState([]);
  const [visibleRows, setVisibleRows] = useState(100);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [, setFileType] = useState('');
  
  // Prediction state
  const [predictions, setPredictions] = useState([]);
  const [predicting, setPredicting] = useState(false);
  const [showPredictions, setShowPredictions] = useState(false);
  
  // CRUD state
  const [editingIndex, setEditingIndex] = useState(null);
  const [editingData, setEditingData] = useState({});
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [newRowData, setNewRowData] = useState({});

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setFileName(selectedFile.name);
    setError('');

    // Check file extension to determine type
    const extension = selectedFile.name.split('.').pop().toLowerCase();
    setFileType(extension);

    setLoading(true);

    if (['csv', 'txt'].includes(extension)) {
      // Parse CSV
      Papa.parse(selectedFile, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          handleParseComplete(results.data, Object.keys(results.data[0] || {}));
        },
        error: (error) => {
          setError(`Error parsing CSV: ${error.message}`);
          setLoading(false);
        }
      });
    } else if (['xlsx', 'xls'].includes(extension)) {
      // Parse Excel
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          const firstSheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheetName];
          const jsonData = XLSX.utils.sheet_to_json(worksheet);
          handleParseComplete(jsonData, Object.keys(jsonData[0] || {}));
        } catch (error) {
          setError(`Error parsing Excel: ${error.message}`);
          setLoading(false);
        }
      };
      reader.onerror = () => {
        setError('Error reading file');
        setLoading(false);
      };
      reader.readAsArrayBuffer(selectedFile);
    } else {
      setError('Unsupported file format. Please upload a CSV or Excel file.');
      setLoading(false);
      setFile(null);
      setFileName('');
    }
  };

  const handleParseComplete = (parsedData, parsedHeaders) => {
    setAllData(parsedData);
    setHeaders(parsedHeaders);
    setData(parsedData.slice(0, 100));
    setVisibleRows(100);
    setLoading(false);
    
    // Initialize newRowData with empty values for each header
    const emptyRow = {};
    parsedHeaders.forEach(header => {
      emptyRow[header] = '';
    });
    setNewRowData(emptyRow);
  };
  const handleClearFile = () => {
    setFile(null);
    setFileName('');
    setHeaders([]);
    setData([]);
    setAllData([]);
    setVisibleRows(100);
    setError('');
    setEditingIndex(null);
    setEditingData({});
    setShowAddDialog(false);
    setNewRowData({});
    setPredictions([]);
    setShowPredictions(false);
  };

  const handleShowMore = () => {
    if (visibleRows < allData.length) {
      const nextChunk = Math.min(visibleRows + 100, allData.length);
      setData(allData.slice(0, nextChunk));
      setVisibleRows(nextChunk);
    }
  };
  
  // Edit row functions
  const handleEditStart = (rowIndex) => {
    setEditingIndex(rowIndex);
    setEditingData({...data[rowIndex]});
  };
  
  const handleEditCancel = () => {
    setEditingIndex(null);
    setEditingData({});
  };
  
  const handleEditSave = () => {
    if (editingIndex !== null) {
      // Update visible data
      const newData = [...data];
      newData[editingIndex] = editingData;
      setData(newData);
      
      // Update all data
      const globalIndex = editingIndex < visibleRows ? editingIndex : visibleRows + (editingIndex - visibleRows);
      const newAllData = [...allData];
      newAllData[globalIndex] = editingData;
      setAllData(newAllData);
      
      // Exit edit mode
      setEditingIndex(null);
      setEditingData({});
    }
  };
  
  const handleEditFieldChange = (header, value) => {
    setEditingData(prev => ({
      ...prev,
      [header]: value
    }));
  };
  
  // Delete row functions
  const handleDeleteRow = (rowIndex) => {
    // Update visible data
    const newData = [...data];
    newData.splice(rowIndex, 1);
    setData(newData);
    
    // Update all data
    const globalIndex = rowIndex < visibleRows ? rowIndex : visibleRows + (rowIndex - visibleRows);
    const newAllData = [...allData];
    newAllData.splice(globalIndex, 1);
    setAllData(newAllData);
  };
  
  // Add row functions
  const handleAddRowStart = () => {
    setShowAddDialog(true);
  };
  
  const handleAddRowCancel = () => {
    setShowAddDialog(false);
    // Reset form
    const emptyRow = {};
    headers.forEach(header => {
      emptyRow[header] = '';
    });
    setNewRowData(emptyRow);
  };
  
  const handleAddRowSave = () => {
    // Add to all data at the beginning
    const newAllData = [newRowData, ...allData];
    setAllData(newAllData);
    
    // Update visible data if needed
    if (visibleRows >= 100) {
      // If we're showing 100+ rows, add the new row to visible data
      const newVisibleData = [newRowData, ...data.slice(0, 99)];
      setData(newVisibleData);
    } else {
      // If we're showing fewer than 100 rows, just add it to the visible data
      const newVisibleData = [newRowData, ...data];
      setData(newVisibleData);
    }
    
    // Close dialog and reset form
    setShowAddDialog(false);
    const emptyRow = {};
    headers.forEach(header => {
      emptyRow[header] = '';
    });
    setNewRowData(emptyRow);
  };
    const handleAddFieldChange = (header, value) => {
    setNewRowData(prev => ({
      ...prev,
      [header]: value
    }));
  };

  // Prediction functions
  const handlePredict = async () => {
    if (allData.length === 0) {
      setError('No data to predict');
      return;
    }

    setPredicting(true);
    setError('');

    try {
      const response = await axios.post('http://localhost:5000/predict', allData);
      
      if (response.data.predictions) {
        setPredictions(response.data.predictions);
        setShowPredictions(true);
      } else {
        setError('Invalid response from prediction service');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setError('Failed to get predictions. Make sure the backend server is running.');
    } finally {
      setPredicting(false);
    }
  };

  const getDataWithPredictions = () => {
    if (!showPredictions || predictions.length === 0) {
      return allData;
    }

    return allData.map((row, index) => ({
      ...row,
      PredictionResult: predictions[index]?.prediction || 'N/A',
      PredictionConfidence: predictions[index]?.confidence 
        ? `${(predictions[index].confidence * 100).toFixed(1)}%` 
        : 'N/A'
    }));
  };

  const getHeadersWithPredictions = () => {
    if (!showPredictions) {
      return headers;
    }
    return [...headers, 'PredictionResult', 'PredictionConfidence'];
  };

  const getPredictionChip = (prediction) => {
    if (prediction === 'Approved') {
      return <Chip label="Approved" color="success" size="small" />;
    } else if (prediction === 'Rejected') {
      return <Chip label="Rejected" color="error" size="small" />;
    } else {
      return <Chip label="N/A" color="default" size="small" />;
    }
  };
  const handleExportToExcel = () => {
    if (allData.length === 0) {
      setError('No data to export');
      return;
    }
  
    try {
      const dataToExport = getDataWithPredictions();
      const headersToExport = getHeadersWithPredictions();
      
      // Create worksheet
      const ws = XLSX.utils.json_to_sheet(dataToExport, { header: headersToExport });
      
      // Create workbook
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, 'Loan Data');
      
      // Generate filename
      const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
      const filename = fileName 
        ? `${fileName.split('.')[0]}_with_predictions_${timestamp}.xlsx` 
        : `loan_data_with_predictions_${timestamp}.xlsx`;
      
      // Download file
      XLSX.writeFile(wb, filename);
    } catch (error) {
      console.error('Export failed:', error);
      setError('Failed to export data');
    }
  };

  return (
    <Card sx={{ 
      maxWidth: '100%', 
      mx: 'auto', 
      mt: 4, 
      boxShadow: 3
    }}>
      <CardContent>
        <Typography variant="h5" gutterBottom sx={{ textAlign: 'center' }}>
          Upload Transaction Data
        </Typography>
        
        {!file ? (
          <Box sx={{ 
            border: '2px dashed #ccc', 
            borderRadius: 2, 
            p: 4,
            textAlign: 'center',
            mb: 3,
            backgroundColor: '#f8f8f8',
            minHeight: 200,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center'
          }}>
            <input
              accept=".csv,.xlsx,.xls,.txt"
              type="file"
              id="csv-upload"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="csv-upload">
              <Button
                variant="contained"
                component="span"
                startIcon={<CloudUploadIcon />}
                sx={{ mb: 2 }}
              >
                Select File
              </Button>
            </label>
            <Typography variant="body2" color="textSecondary">
              Accepted formats: CSV, Excel (.xlsx, .xls)
            </Typography>
          </Box>
        ) : (
          <>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                    {fileName} {loading && <CircularProgress size={20} sx={{ ml: 1 }} />}
                </Typography>                <Box>
                    <Button
                        startIcon={<DownloadIcon />}
                        variant="outlined"
                        color="primary"
                        sx={{ mr: 1 }}
                        onClick={handleExportToExcel}
                        disabled={allData.length === 0}
                    >
                        Export Excel
                    </Button>
                    <Button
                        startIcon={<PredictIcon />}
                        variant="contained"
                        color="secondary"
                        sx={{ mr: 1 }}
                        onClick={handlePredict}
                        disabled={allData.length === 0 || predicting}
                    >
                        {predicting ? 'Predicting...' : 'Predict Loans'}
                    </Button>
                    <Button
                        startIcon={<AddIcon />}
                        variant="contained"
                        color="primary"
                        sx={{ mr: 1 }}
                        onClick={handleAddRowStart}
                    >
                        Add Row
                    </Button>
                    <IconButton onClick={handleClearFile}>
                        <DeleteIcon />
                    </IconButton>
                </Box>
                </Box>
              {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <>
                {predicting && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    Making predictions... Please wait.
                  </Alert>
                )}
                
                {showPredictions && (
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Predictions completed! Results are shown in the table and can be exported.
                  </Alert>
                )}

                {headers.length > 0 && (
                  <>                    <TableContainer component={Paper} sx={{ maxHeight: 500, mb: 2 }}>
                      <Table stickyHeader size="small">
                        <TableHead>
                          <TableRow>
                            {getHeadersWithPredictions().map((header, index) => (
                              <TableCell key={index} sx={{ fontWeight: 'bold' }}>
                                {header}
                              </TableCell>
                            ))}
                            <TableCell sx={{ fontWeight: 'bold', width: 100 }}>
                              Actions
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {data.map((row, rowIndex) => {
                            const rowWithPredictions = showPredictions && predictions[rowIndex] 
                              ? {
                                  ...row,
                                  PredictionResult: predictions[rowIndex]?.prediction || 'N/A',
                                  PredictionConfidence: predictions[rowIndex]?.confidence 
                                    ? `${(predictions[rowIndex].confidence * 100).toFixed(1)}%` 
                                    : 'N/A'
                                }
                              : row;

                            return (
                              <TableRow key={rowIndex} hover>
                                {getHeadersWithPredictions().map((header, colIndex) => (
                                  <TableCell key={`${rowIndex}-${colIndex}`}>
                                    {editingIndex === rowIndex && !['PredictionResult', 'PredictionConfidence'].includes(header) ? (
                                      <TextField
                                        value={editingData[header] || ''}
                                        onChange={(e) => handleEditFieldChange(header, e.target.value)}
                                        size="small"
                                        fullWidth
                                      />
                                    ) : header === 'PredictionResult' && showPredictions ? (
                                      getPredictionChip(rowWithPredictions[header])
                                    ) : (
                                      rowWithPredictions[header]
                                    )}
                                  </TableCell>
                                ))}
                                <TableCell>
                                  {editingIndex === rowIndex ? (
                                    <>
                                      <Tooltip title="Save">
                                        <IconButton 
                                          color="primary" 
                                          size="small"
                                          onClick={handleEditSave}
                                        >
                                          <SaveIcon />
                                        </IconButton>
                                      </Tooltip>
                                      <Tooltip title="Cancel">
                                        <IconButton 
                                          color="default" 
                                          size="small"
                                          onClick={handleEditCancel}
                                        >
                                          <CancelIcon />
                                        </IconButton>
                                      </Tooltip>
                                    </>
                                  ) : (
                                    <>
                                      <Tooltip title="Edit">
                                        <IconButton 
                                          color="primary" 
                                          size="small"
                                          onClick={() => handleEditStart(rowIndex)}
                                        >
                                          <EditIcon />
                                        </IconButton>
                                      </Tooltip>
                                      <Tooltip title="Delete">
                                        <IconButton 
                                          color="error" 
                                          size="small"
                                          onClick={() => handleDeleteRow(rowIndex)}
                                        >
                                          <DeleteIcon />
                                        </IconButton>
                                      </Tooltip>
                                    </>
                                  )}
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" color="textSecondary">
                        Showing {data.length} of {allData.length} rows
                      </Typography>
                      
                      {visibleRows < allData.length && (
                        <Button 
                          onClick={handleShowMore}
                          variant="outlined"
                        >
                          Show More
                        </Button>
                      )}
                    </Box>
                  </>
                )}
              </>
            )}
            
            {/* Add Row Dialog */}
            <Dialog 
              open={showAddDialog} 
              onClose={handleAddRowCancel}
              fullWidth
              maxWidth="md"
            >
              <DialogTitle>Add New Row</DialogTitle>
              <DialogContent>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2, my: 2 }}>
                  {headers.map((header, index) => (
                    <TextField
                      key={index}
                      label={header}
                      value={newRowData[header] || ''}
                      onChange={(e) => handleAddFieldChange(header, e.target.value)}
                      fullWidth
                      margin="dense"
                    />
                  ))}
                </Box>
              </DialogContent>
              <DialogActions>
                <Button onClick={handleAddRowCancel} color="primary">
                  Cancel
                </Button>
                <Button 
                  onClick={handleAddRowSave} 
                  variant="contained" 
                  color="primary"
                  startIcon={<AddIcon />}
                >
                  Add Row
                </Button>
              </DialogActions>
            </Dialog>
          </>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default CSVUpload;