#include "settingsdialog.h"
#include <QComboBox>
#include <QFormLayout>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>

SettingsDialog::SettingsDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("System Settings");
    setMinimumSize(300, 150);

    auto *layout = new QFormLayout(this);

    // Initialize theme selection ComboBox
    _themeCombo = new QComboBox(this);
    _themeCombo->addItem("Light", "light");
    _themeCombo->addItem("Dark", "dark");
    _themeCombo->addItem("System Default", "system");

    layout->addRow("Theme Mode:", _themeCombo);

    // Initialize camera source selection
    _cameraSourceCombo = new QComboBox(this);
    _cameraSourceCombo->addItem("TMVisionary (TCP)", 0);
    _cameraSourceCombo->addItem("Intel RealSense (USB)", 1);
    
    layout->addRow("Camera Source:", _cameraSourceCombo);

    // Camera TCP Offset
    auto *offsetLayout = new QHBoxLayout();
    _camOffsetX = new QDoubleSpinBox(this);
    _camOffsetY = new QDoubleSpinBox(this);
    _camOffsetZ = new QDoubleSpinBox(this);
    
    for (auto *spin : {_camOffsetX, _camOffsetY, _camOffsetZ}) {
        spin->setRange(-1000.0, 1000.0);
        spin->setDecimals(1);
        spin->setSuffix(" mm");
    }

    offsetLayout->addWidget(new QLabel("X:"));
    offsetLayout->addWidget(_camOffsetX);
    offsetLayout->addWidget(new QLabel("Y:"));
    offsetLayout->addWidget(_camOffsetY);
    offsetLayout->addWidget(new QLabel("Z:"));
    offsetLayout->addWidget(_camOffsetZ);

    layout->addRow("Camera TCP Offset:", offsetLayout);

    // Save settings button
    auto *saveBtn = new QPushButton("Save Changes", this);
    saveBtn->setMinimumHeight(35); // Make the button slightly larger for easier clicking

    connect(saveBtn, &QPushButton::clicked, this, &SettingsDialog::accept);

    layout->addRow(saveBtn);
}

QString SettingsDialog::getSelectedTheme() const
{
    return _themeCombo->currentData().toString();
}

int SettingsDialog::getSelectedCameraSource() const
{
    return _cameraSourceCombo->currentData().toInt();
}

void SettingsDialog::setCameraOffset(double x, double y, double z)
{
    _camOffsetX->setValue(x);
    _camOffsetY->setValue(y);
    _camOffsetZ->setValue(z);
}

void SettingsDialog::getCameraOffset(double &x, double &y, double &z) const
{
    x = _camOffsetX->value();
    y = _camOffsetY->value();
    z = _camOffsetZ->value();
}