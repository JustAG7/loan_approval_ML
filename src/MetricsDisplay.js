import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Divider
} from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TargetIcon from '@mui/icons-material/MyLocation';
import BalanceIcon from '@mui/icons-material/Balance';

const MetricsDisplay = ({ metrics }) => {
  if (!metrics || !metrics.has_labels) {
    return null;
  }

  const { accuracy, precision, recall, f1_score, confusion_matrix, classification_report } = metrics;

  // Format percentage values
  const formatPercentage = (value) => `${(value * 100).toFixed(2)}%`;

  // Create confusion matrix display data
  const confusionMatrixData = [
    { actual: 'Rejected (0)', predicted_0: confusion_matrix[0][0], predicted_1: confusion_matrix[0][1] },
    { actual: 'Approved (1)', predicted_0: confusion_matrix[1][0], predicted_1: confusion_matrix[1][1] }
  ];

  // Extract class-specific metrics
  const class0Metrics = classification_report['0'] || {};
  const class1Metrics = classification_report['1'] || {};

  return (
    <Card sx={{ mt: 3, boxShadow: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <AssessmentIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h5" component="h2">
            Model Performance Metrics
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {/* Overall Metrics Cards */}
          <Grid item xs={12} md={6}>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                Overall Performance
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Card variant="outlined" sx={{ p: 2, textAlign: 'center', bgcolor: 'success.50' }}>
                    <TrendingUpIcon sx={{ color: 'success.main', mb: 1 }} />
                    <Typography variant="body2" color="textSecondary">
                      Accuracy
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {formatPercentage(accuracy)}
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined" sx={{ p: 2, textAlign: 'center', bgcolor: 'info.50' }}>
                    <TargetIcon sx={{ color: 'info.main', mb: 1 }} />
                    <Typography variant="body2" color="textSecondary">
                      Precision
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {formatPercentage(precision)}
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined" sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.50' }}>
                    <BalanceIcon sx={{ color: 'warning.main', mb: 1 }} />
                    <Typography variant="body2" color="textSecondary">
                      Recall
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {formatPercentage(recall)}
                    </Typography>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined" sx={{ p: 2, textAlign: 'center', bgcolor: 'secondary.50' }}>
                    <AssessmentIcon sx={{ color: 'secondary.main', mb: 1 }} />
                    <Typography variant="body2" color="textSecondary">
                      F1-Score
                    </Typography>
                    <Typography variant="h6" fontWeight="bold">
                      {formatPercentage(f1_score)}
                    </Typography>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          </Grid>

          {/* Confusion Matrix */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              Confusion Matrix
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold', bgcolor: 'grey.100' }}>
                      Actual / Predicted
                    </TableCell>
                    <TableCell align="center" sx={{ fontWeight: 'bold', bgcolor: 'grey.100' }}>
                      Rejected (0)
                    </TableCell>
                    <TableCell align="center" sx={{ fontWeight: 'bold', bgcolor: 'grey.100' }}>
                      Approved (1)
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {confusionMatrixData.map((row, index) => (
                    <TableRow key={index}>
                      <TableCell sx={{ fontWeight: 'bold' }}>
                        {row.actual}
                      </TableCell>
                      <TableCell align="center" sx={{ 
                        bgcolor: index === 0 ? 'success.50' : 'error.50',
                        fontWeight: 'bold'
                      }}>
                        {row.predicted_0}
                      </TableCell>
                      <TableCell align="center" sx={{ 
                        bgcolor: index === 1 ? 'success.50' : 'error.50',
                        fontWeight: 'bold'
                      }}>
                        {row.predicted_1}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>

          {/* Class-wise Performance */}
          <Grid item xs={12}>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              Class-wise Performance
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Chip label="Rejected (Class 0)" color="error" size="small" sx={{ mr: 1 }} />
                    <Typography variant="subtitle1" fontWeight="bold">
                      Rejection Performance
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Precision:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatPercentage(class0Metrics.precision || 0)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Recall:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatPercentage(class0Metrics.recall || 0)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">F1-Score:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatPercentage(class0Metrics['f1-score'] || 0)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Support:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {class0Metrics.support || 0}
                    </Typography>
                  </Box>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card variant="outlined" sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Chip label="Approved (Class 1)" color="success" size="small" sx={{ mr: 1 }} />
                    <Typography variant="subtitle1" fontWeight="bold">
                      Approval Performance
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Precision:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatPercentage(class1Metrics.precision || 0)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Recall:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatPercentage(class1Metrics.recall || 0)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">F1-Score:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {formatPercentage(class1Metrics['f1-score'] || 0)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Support:</Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {class1Metrics.support || 0}
                    </Typography>
                  </Box>
                </Card>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default MetricsDisplay;
