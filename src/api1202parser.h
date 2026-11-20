#ifndef API1202PARSER_H
#define API1202PARSER_H

#include "types.h"
#include <QByteArray>

class Api1202Parser
{
public:
    static bool parse(const QByteArray &jsonBytes, MarkerFrame1202 &outFrame, QString &err);
};

#endif
