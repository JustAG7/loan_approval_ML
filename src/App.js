import React from 'react';
import { CssBaseline, ThemeProvider, createTheme, AppBar, Toolbar, Typography } from '@mui/material';
import CreditCardIcon from '@mui/icons-material/CreditCard';
import CSVUpload from './CSVUpload';

// Enhanced theme with better colors
const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f5'
    }
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" sx={{ mb: 4 }}>
        <Toolbar>
          <CreditCardIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div">
            Loan Approval Prediction
          </Typography>
        </Toolbar>
      </AppBar>
        <CSVUpload />
    </ThemeProvider>
  );
}

export default App;