#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>

// Dùng Forward Declaration giúp file header nhẹ hơn
class QComboBox;
class QDoubleSpinBox;

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = nullptr);
    ~SettingsDialog() override = default;

    QString getSelectedTheme() const;
    int getSelectedCameraSource() const;
    void setCameraOffset(double x, double y, double z);
    void getCameraOffset(double &x, double &y, double &z) const;

private:
    QComboBox      *_themeCombo = nullptr;
    QComboBox      *_cameraSourceCombo = nullptr;
    QDoubleSpinBox *_camOffsetX = nullptr;
    QDoubleSpinBox *_camOffsetY = nullptr;
    QDoubleSpinBox *_camOffsetZ = nullptr;
};

#endif // SETTINGSDIALOG_H